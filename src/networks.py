import torch.nn.init as init
from net_utils import *
import math


class MVFF(nn.Module):

    def __init__(self,device,input_dims, num_classes, hidden_dim, n_view=3):
        super(MVFF, self).__init__()
        self.device = device
        self.n_view = n_view
        self.sub_nets = nn.ModuleList([DeepCALayer(input_dim=input_dims[i], hidden_dim=hidden_dim)
                                       for i in range(n_view)])

        self.common_project_1 = nn.Sequential(
            nn.Conv1d(hidden_dim * n_view, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.common_project_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.cross_linear = ChannelGate(k_size=7)

        self.co_layers = nn.ModuleList([CoAttention(hidden_dim) for _ in range(n_view)])

        # self.w = nn.Parameter(torch.ones(n_view))

        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc_out = nn.Linear(hidden_dim * n_view, num_classes)

        self.init_weights()

    def forward(self, data):
        x = [data.x, data.x1, data.x2]
        # phase 1 compute Specific Feature
        enc_outs = []
        for i, enc_layer in enumerate(self.sub_nets):
            conv_attn = enc_layer(x[i])
            enc_outs.append(conv_attn)
        # out 1
        # phase 2.2 compute shared feats
        common_feat_ini = self.common_project_1(torch.cat(enc_outs, dim=1))
        common_feat_tep = self.common_project_2(common_feat_ini)

        common_feat = common_feat_tep * self.cross_linear(common_feat_tep) + common_feat_ini
        # phase 2 compute shared feature
        attn_list = []
        for co_layer, sub_x in zip(self.co_layers, enc_outs):
            enc_attn = co_layer(common_feat, sub_x)
            attn_list.append(enc_attn)

        attn_feat = torch.cat(attn_list, dim=1)


        # weight_var = [torch.exp(self.w[i]) / torch.sum(torch.exp(self.w)) for i in range(self.n_view)]
        # attn_feat = None
        # for i in range(self.n_view):
        #     temp_feat = weight_var[i] * high_level_feats[i]
        #     if attn_feat is None:
        #         attn_feat = temp_feat
        #     else:
        #         attn_feat += temp_feat

        high_level_feat = self.maxpool(attn_feat).flatten(1)
        y = self.fc_out(high_level_feat)
        y = torch.sigmoid(y)
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class CoAttention(nn.Module):
    def __init__(self,hidden_dim,k_size=3):
        super(CoAttention, self).__init__()
        self.hidden_dim = hidden_dim

        self.compress_shared = ChannelPool()
        self.compress_ini = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        )

    def forward(self, shared_feat, sub_feat):
        # cross scaled Attention
        shared_spatial_global = self.compress_shared(shared_feat) # (B,1,N)
        sub_spatial_global = self.compress_ini(sub_feat)  # (B,1,N)
        co_spatial_global = torch.cat([shared_spatial_global, sub_spatial_global], dim=1)
        co_spatial_global = self.spatial(co_spatial_global)
        sim_scale = torch.sigmoid(co_spatial_global)
        sub_state = sub_feat * sim_scale.expand_as(sub_feat)

        return sub_state + sub_feat


class DeepCNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_classes):
        super(DeepCNN, self).__init__()

        self.encoder = DeepCALayer(input_dim,hidden_dim)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def forward(self,data):
        x = data.x
        enc_x = self.encoder(x)
        high_level_feat = self.maxpool(enc_x).flatten(1)

        output = self.fc_out(high_level_feat)
        output = torch.sigmoid(output)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)