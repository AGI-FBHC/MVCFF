import os
import pickle
from tqdm import tqdm
from collections import Counter
import click as ck
from sklearn import preprocessing
from goatools.obo_parser import GODag
from goatools.anno.idtogos_reader import IdToGosReader
from goatools.semantic import TermCounts, get_info_content,lin_sim,resnik_sim
import pandas as pd
import itertools
import numpy as np
from scipy.io import savemat
import networkx as nx
from sklearn.preprocessing import StandardScaler
import sys  # 导入sys模块
sys.setrecursionlimit(20000)  # 将默认的递归深度修改为3000
import matplotlib.pyplot as plt

@ck.command()
@ck.option(
    '--datasets-name', '-dname', default='homo',
    help='Test data file')
@ck.option(
    '--ont', '-o', default='mf',
    help='GO subontology (bp, mf, cc)')
def main(datasets_name,ont):
    # 1 . 根据每个术语建立其标签子图
    F_txt = open(f'data/{datasets_name}/{ont.upper()}_label_graph_record.txt','a+')
    print(f"DataSets:{datasets_name}\tOnt:{ont}")

    high_confidence_alpha = 0.5
    binary_condition_alpha = 0.9
    binary_knowledge_alpha = 0.5
    is_random_walk = True
    min_max_scaler = preprocessing.MinMaxScaler()
    godag = GODag(f'data/{datasets_name}/go-basic.obo', optional_attrs={'relationship'})
    df = pd.read_pickle(f'data/{datasets_name}/{ont}/train_df.pkl')
    annotations = df['prop_annotations'].values
    associations_file = f"data/{datasets_name}/{ont}/{ont}_associations.tab"
    associations = pd.DataFrame(df, columns=["proteins", "prop_annotations"])
    # pandas 逐行写入 associations
    with open(associations_file, "w") as f:
        f.write("Entry\tGene ontology IDs")
        f.write("\n")
        for i, row in enumerate(associations.itertuples()):
            f.write("{}\t".format(row.proteins))
            for i, go_id in enumerate(row.prop_annotations):
                f.write("{};".format(go_id))
            f.write("\n")

    associations = IdToGosReader(f"data/{datasets_name}/{ont}/{ont}_associations.tab", godag=godag).get_id2gos('all')
    termcounts = TermCounts(godag, associations)
    terms = np.loadtxt(f'data/{datasets_name}/{ont}/{ont}.txt', dtype=str)
    # vex
    index_dict = {i: v for i, v in enumerate(terms)}
    num_classes = len(terms)
    print(num_classes)
    # 统计图构建
    node_weights = get_nodes_score(annotations,terms)

    is_edge_weights,Sim_edge_list = get_edge_list(godag,termcounts,terms,sim_method='LinSim')   # 'LinSim'    'ResnikSim'
    print(len(Sim_edge_list))

    if is_edge_weights:
        edge_list = pd.DataFrame(columns=['node1', 'node2'])
        for i in range(len(Sim_edge_list)):
            row = Sim_edge_list.iloc[i]
            if row.weight.item() > high_confidence_alpha:
                tmps = pd.Series({'node1': row.node1, 'node2': row.node2})
                edge_list = edge_list.append(tmps, ignore_index=True)

        is_edge_weights = False
    print(len(edge_list))

    # create a vertex index
    node1_unique = edge_list['node1'].unique()
    node2_unique = edge_list['node2'].unique()
    all_nodes_unique = list(set(node1_unique).union(set(node2_unique)))
    # list must be sorted because union will return a different order of nodes every time
    all_nodes_unique.sort()
    all_nodes_unique_df = pd.DataFrame(
        {"node": all_nodes_unique, "node_index": range(0, len(all_nodes_unique))})

    # 统计哪些术语不在图中，以及多了哪些术语
    both_terms = set(all_nodes_unique_df['node'].values.tolist()) & set(terms)
    add_terms = set(all_nodes_unique_df['node'].values.tolist()) - set(terms)
    removed_terms = set(terms) - set(all_nodes_unique_df['node'].values.tolist())
    print("intersection:", len(both_terms))
    print("Added:", add_terms)
    print("Removed:", removed_terms)

    # create an edge list using the vertex index
    node1_in_vertex_index = pd.merge(edge_list, all_nodes_unique_df, left_on=['node1'], right_on=['node'], how='left')
    node1_vertex_index = node1_in_vertex_index['node_index'].tolist()

    node2_in_vertex_index = pd.merge(edge_list, all_nodes_unique_df, left_on=['node2'], right_on=['node'], how='left')
    node2_vertex_index = node2_in_vertex_index['node_index'].tolist()

    edge_list_vertex_index = pd.DataFrame({"node1": node1_vertex_index, "node2": node2_vertex_index})
    # if is_edge_weights:
    #     # add the weights
    #     edge_list_vertex_index['weight'] = edge_list['weight']

    G_connected, vertex_index = edges_to_net(edge_list_vertex_index, all_nodes_unique_df, is_edge_weights)
    nodes_order = sorted(G_connected.nodes())
    # get the adjacency matrix of the graph, ordered accodring to the nodes
    adj_mat = nx.adjacency_matrix(G_connected, nodelist=nodes_order)
    adj_mat_dense = adj_mat.todense()

    # get the random walk matrix
    if is_random_walk:
        node_deg = G_connected.degree()
        node_deg_order = np.array([node_deg[n] for n in nodes_order])
        node_core = nx.core_number(G_connected)
        node_core_order = np.array([node_core[n] for n in nodes_order])
        norm_adj_mat, random_walk_mat = get_random_walk_matrix('core_norm', is_edge_weights, adj_mat_dense, node_deg_order,
                                                             node_core_order, 0.5)
        adj_mat_dense = min_max_scaler.fit_transform(random_walk_mat)
    nodes = vertex_index.set_index('node')['node_index'].to_dict()  # go_id: id
    # nodes_index = {value: key for key, value in nodes.items()}      # id : go_id
    # weights_sorted, weights_mat = process_weights(node_weights, vertex_index,nodes_order)


    # 将 图的邻接矩阵转化为等大小的相关矩阵
    knowledge_graph_adj = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        p_id = index_dict[i]
        if p_id in nodes:
            for j in range(num_classes):
                s_id = index_dict[j]
                if s_id in nodes:
                    knowledge_graph_adj[i, j] = adj_mat_dense[nodes[p_id], [nodes[s_id]]]

    Conditional_graph = get_co_occurrence_graph(terms, df)
    binary_condition_adj = get_binary_adj(Conditional_graph,binary_condition_alpha)
    binary_knowledge_adj = get_binary_adj(knowledge_graph_adj,binary_knowledge_alpha)
    final_adj = binary_condition_adj + binary_knowledge_adj

    final_adj[final_adj > 1] = 1

    print("Conditional Graph",np.count_nonzero(binary_condition_adj))
    print("Knowledged Graph",np.count_nonzero(binary_knowledge_adj))
    print("Final Graph",np.count_nonzero(final_adj))
    print(f'high_confidence_alpha:{high_confidence_alpha}\tconditional_alpha:{binary_condition_alpha}\tknowledge_alpha:{binary_knowledge_alpha}\tfinal_edges:{np.count_nonzero(final_adj)}',file=F_txt)
    #
    # # 标签采用 onehot 特征
    gox_adj = np.identity(num_classes,np.float32)
    np.save(f'data/{datasets_name}/{ont}/goxfile.npy',gox_adj)
    savemat(f"data/{datasets_name}/{ont}/CondSim.mat", {"A": final_adj})

def get_binary_adj(adj, alpha):
    binary_adj = np.zeros(adj.shape)
    (height, width) = adj.shape
    for i in range(height):
        for j in range(width):
            if adj[i][j] < alpha:
                binary_adj[i][j] = 0
            else:
                binary_adj[i][j] = 1
    return binary_adj
def get_co_occurrence_graph(terms,df):
    terms_dict = {v: i for i, v in enumerate(terms)}
    index_dict = {i: v for i, v in enumerate(terms)}
    num_classes = len(terms)
    Statistic_graph = np.zeros((num_classes, num_classes), dtype=np.int32)
    cnt = Counter()
    for id, row in enumerate(df.itertuples()):
        labels = row.prop_annotations
        new_labels = []
        for go_id in labels:
            if go_id in terms_dict:
                cnt[go_id] += 1
                new_labels.append(go_id)

        label_pairs = list(itertools.permutations(new_labels, 2))
        for pair in label_pairs:
            Statistic_graph[terms_dict[pair[0]], terms_dict[pair[1]]] += 1

    Conditional_graph = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        num_count = cnt[index_dict[i]]
        for j in range(num_classes):
            Conditional_graph[i, j] = Statistic_graph[i, j] / num_count

    return Conditional_graph

def get_nodes_score(annotations,terms):
    cnt = Counter()
    max_n = 0
    for x in annotations:
        cnt.update(x)
    max_n = cnt.most_common(1)[0][1]

    scores = {}
    node_weights = pd.DataFrame(columns=['node', 'weight'])
    for go_id, n in cnt.items():
        if go_id in terms:
            score = n / max_n
            tmps = pd.Series(
                {'node': go_id, 'weight': score})
            node_weights = node_weights.append(tmps, ignore_index=True)
    return node_weights

def get_edge_list(godag,termcounts,terms,sim_method=None):
    if sim_method:
        is_edge_weights = True
        edge_list = pd.DataFrame(columns=['node1', 'node2', 'weight'])
        for go_id in terms:
            # print(go_id)
            Term = godag.query_term(go_id)
            temp_edges = Term.get_all_parent_edges()
            for (p_id, c_id) in temp_edges:
                if sim_method:
                    if sim_method == 'LinSim':
                        sim_value = lin_sim(p_id, c_id, godag, termcounts)
                    elif sim_method == 'ResnikSim':
                        sim_value = resnik_sim(p_id, c_id, godag, termcounts)

                    if sim_value != None and sim_value != 0:
                        tmps = pd.Series(
                            {'node1': p_id, 'node2': c_id, 'weight': sim_value})
                        if p_id in terms and c_id in terms:
                            edge_list = edge_list.append(tmps, ignore_index=True)

    else:
        is_edge_weights = False
        edge_list = pd.DataFrame(columns=['node1', 'node2'])
        for go_id in terms:
            Term = godag.query_term(go_id)
            temp_edges = Term.get_all_parent_edges()
            for (p_id, c_id) in temp_edges:
                    tmps = pd.Series({'node1': p_id, 'node2': c_id})
                    if p_id in terms and c_id in terms:
                        edge_list = edge_list.append(tmps, ignore_index=True)

    return is_edge_weights,edge_list


def process_weights(node_weights, vertex_index, nodes_order):
    # process the input weights and give weight 0 to any node in the network that doesn't have an input weight

    print("Processing weights...")

    # merge the weights df with the index df
    node_weights_index = pd.merge(node_weights, vertex_index, on=['node'], how='outer')
    print(node_weights_index)

    # remove nodes that have a weight but not an index in the network
    if node_weights_index['node_index'].isnull().values.any():
        nodes_with_weight_but_no_index = node_weights_index.loc[node_weights_index['node_index'].isnull(),]
        print(str(nodes_with_weight_but_no_index.shape[
                      0]) + " nodes with an input weight are not in the network -> They will be removed from further analysis!")
        node_weights_index.drop(nodes_with_weight_but_no_index.index.tolist(), inplace=True)

    # give weight 0 to nodes in the network that don't have an input weight
    if node_weights_index['weight'].isnull().values.any():
        nodes_with_index_but_no_weight = node_weights_index.loc[node_weights_index['weight'].isnull(),]
        print(str(nodes_with_index_but_no_weight.shape[
                      0]) + " nodes in the network don't have an input weight -> They will be assigned a weight of 0!")
        node_weights_index.loc[node_weights_index['weight'].isnull(), 'weight'] = 0.0

    # convert index to int
    node_weights_index.loc[:, 'node_index'] = node_weights_index['node_index'].apply(lambda x: int(x))

    # create a node and weight list
    nodes = node_weights_index.iloc[:]['node_index'].tolist()
    weights = node_weights_index.iloc[:]['weight'].tolist()

    # create a dict for the weights
    node_weight_dict = dict(zip(nodes, weights))

    # order the weights of the nodes according to the nodes order
    weights_sorted = [node_weight_dict[node] for node in nodes_order]

    # create a diagonal matrix of the weights  对角矩阵
    weights_mat = np.diag(weights_sorted)

    return weights_sorted, weights_mat

if __name__ == '__main__':
    # LinSim    ResnikSim
    main()








