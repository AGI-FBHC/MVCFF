import numpy as np
import torch
from torch import nn
from torch.nn import init
class Feature_Extract(nn.Module):
    def __init__(self, input_dim=21, num_filters=8 * [256], filter_sizes=list(range(8, 129, 8)), num_classes=256):
        super(Feature_Extract, self).__init__()

        # Define 1D convolutional layers
        cnn_layers = [
            nn.Conv1d(input_dim, num_filters[i], kernel_size=filter_sizes[i], padding=int(filter_sizes[i] / 2) - 1)
            for i in range(len(num_filters))]
        self.cnn = nn.ModuleList(cnn_layers)

        self.ChannelGate = ChannelGate(num_filters[0], 16, ['avg', 'max'])
        self.SpatialGate = SpatialGate()

        # Define global max pooling
        pool_layers = [nn.AdaptiveMaxPool1d(1) for _ in num_filters]
        self.globalpool = nn.ModuleList(pool_layers)

        # Define fully-connected layers
        self.fc_out = nn.Linear(sum(num_filters), num_classes)

    def forward(self, data):
        # Compute 1D convolutional part and apply global max pooling

        all_x = []
        for cnn_layer, pool_layer in zip(self.cnn, self.globalpool):
            cnn_out = cnn_layer(data)

            cnn_cout = self.ChannelGate(cnn_out)
            cnn_sout = self.SpatialGate(cnn_out)
            att = 1 + torch.sigmoid(cnn_cout * cnn_sout)
            cnn_out = att * cnn_out

            y = pool_layer(cnn_out)

            all_x.append(y)

        # Concatenate all channels and flatten vector
        x = torch.cat(all_x, dim=1)  # (Batch,512*16 , 2)
        x = torch.flatten(x, 1)
        output = self.fc_out(x)

        output = torch.sigmoid(output)

        return output


class DeepMVCAPF(nn.Module):
    def __init__(self, input_dims=[21,128,20,292], num_filters=8 * [512], hidden_size = 256, num_classes=256):
        super(DeepMVCAPF, self).__init__()

        # define the pre-fusion subnetworks

        feature_extract_layers = [
            Feature_Extract(input_dim=input_dims[i],num_filters=num_filters,num_classes=hidden_size)
            for i in range(len(input_dims))]

        self.feature_ectract_layers = nn.ModuleList(feature_extract_layers)

        # define fusion layers
        self.output_linear = nn.Linear(hidden_size * len(input_dims), num_classes)


    def forward(self, data):
        # re_1 适用LMF fusion  , re_2 适用 ACN fusion
        input_datas = [data.x,data.x1,data.x2,data.x3]
        all_x = []
        for id,feature_extract_layer in enumerate(self.feature_ectract_layers):
            feature = feature_extract_layer(input_datas[id])
            all_x.append(feature)


        # concat
        out = torch.cat(all_x,dim=1)
        out = self.output_linear(out)
        out = torch.sigmoid(out)

        return out

# Squeeze-and-Excitation
class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

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

    def forward(self, x):
        b, c, _,= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEConvAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction,kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1,padding=0),
            nn.Sigmoid()
        )

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

    def forward(self, x):
        b, c, _,= x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# class CoordAtt(nn.Module):
#     def __init__(self, ):
#         super(CoordAtt, self).__init__()
#         self.pool_c = nn.AdaptiveAvgPool1d(1)
#         self.pool_n = ChannelPool()
#
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=1,padding=0)
#         self.bn1 = nn.BatchNorm1d(1)
#         self.act = h_swish()
#
#         self.conv_c = nn.Conv1d(1, 1, kernel_size=1, padding=0)
#         self.conv_n = nn.Conv1d(1, 1, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         b,c,n = x.size()
#         x_c = self.pool_c(x).transpose(1,2)
#         x_n = self.pool_n(x)
#         y = torch.cat([x_c, x_n], dim=2)
#
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_c, x_n = torch.split(y, [c, n], dim=2)
#         a_c = self.conv_c(x_c).sigmoid()
#         a_n = self.conv_n(x_n).sigmoid()
#
#         a_c = a_c.transpose(1,2)
#         out = identity * a_c * a_n
#
#         return out


class CoordAtt(nn.Module):
    def __init__(self, kernel_size = 3):
        super(CoordAtt, self).__init__()
        self.max_pool_c = nn.AdaptiveMaxPool1d(1)
        self.avg_pool_c = nn.AdaptiveAvgPool1d(1)
        self.pool_n = ChannelPool()
        self.conv1 = BasicConv(2, 1, kernel_size=kernel_size,  padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        identity = x

        b, c, n  = x.size()

        x_c = torch.cat((self.max_pool_c(x),self.avg_pool_c(x)),dim=2).transpose(1,2)
        x_n =  self.pool_n(x)
        y = torch.cat([x_c, x_n], dim=2)
        y = self.conv1(y)
        x_n, x_c = torch.split(y, [c, n], dim=2)
        x_n = x_n.transpose(1, 2)

        a_c = torch.sigmoid(x_c)
        a_n = torch.sigmoid(x_n)
        x_attn = identity * a_c * a_n

        return x_attn



def conv_2(in_planes,out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False), nn.BatchNorm2d(out_planes),nn.ReLU(inplace=True))


def cov(input):
    b,c,h = input.size()
    x = input - torch.mean(input)
    x = x.view(b*c,h)
    cov_matrix = torch.matmul(x.T,x) / x.shape[0]
    return cov_matrix

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


class CSAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CSAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        c_score = self.ChannelGate(x)
        s_score = self.SpatialGate(x)
        h_score = torch.sigmoid(c_score + s_score)
        x_attn = x * h_score
        return x_attn + x


class SpatialGate(nn.Module):
    def __init__(self,kernel_size = 3):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size=kernel_size,  padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)   # [AvgPool , MaxPool]
        x_out = self.spatial(x_compress)  # batch 1 L
        scale = torch.sigmoid(x_out)   # broadcasting

        return scale

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Conv1d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, padding=0),
        )

        self.pool_types = pool_types
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x)
                # avg_pool = torch.flatten(avg_pool,1)
                channel_att_raw = self.mlp(avg_pool)
                channel_att_raw = torch.flatten(channel_att_raw, 1)
            elif pool_type=='max':
                max_pool = self.maxpool(x)
                # max_pool = torch.flatten(max_pool, 1)
                channel_att_raw = self.mlp(max_pool)
                channel_att_raw = torch.flatten(channel_att_raw, 1)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2)

        return scale

class DeepCNN(nn.Module):
    def __init__(self, input_dim=21, num_filters=8 * [256], r=16,num_classes=256):
        super(DeepCNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, num_filters[0], kernel_size=3, padding=1, ),
                                   nn.BatchNorm1d(num_filters[0]))
        self.conv2 = nn.Sequential(nn.Conv1d(num_filters[0], num_filters[0], kernel_size=3, padding=1, ),
                                   nn.BatchNorm1d(num_filters[0]))
        self.conv3 = nn.Sequential(nn.Conv1d(num_filters[0], num_filters[0], kernel_size=3, padding=1, ),
                                   nn.BatchNorm1d(num_filters[0]))

        self.pool1 = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

        self.attn = CSAM(num_filters[0], r)

        # Define fully-connected layers
        self.fc_out = nn.Linear(num_filters[0], num_classes)

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


    def forward(self, data):
        # Compute 1D convolutional part and apply global max pooling
        x = data.x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = self.attn(conv3)
        conv = self.pool1(conv3)
        x = torch.flatten(conv, 1)
        x = self.relu(x)
        output = self.fc_out(x)
        output = torch.sigmoid(output)

        return output


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,padding,relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)


class Co_attention(nn.Module):
    def __init__(self, hidden_dim,r=4):
        super(Co_attention, self).__init__()
        self.inter_channels = hidden_dim // r
        self.theta_g = nn.Conv1d(hidden_dim, self.inter_channels, kernel_size=3, padding=1)
        self.theta_s = nn.Conv1d(hidden_dim, self.inter_channels, kernel_size=3, padding=1)

        self.softmax = nn.Softmax(1)

        self.rate = nn.Parameter(torch.Tensor(1))
    def forward(self, guided_feat,sub_feat):
        B, C,N = guided_feat.size()

        theta_g = self.theta_g(guided_feat).permute(0, 2, 1).contiguous()
        theta_s = self.theta_s(sub_feat)

        f = torch.matmul(theta_g, theta_s) / N
        f = self.softmax(f)

        non_sub = torch.matmul(sub_feat,f)

        return self.rate * non_sub + sub_feat





class eca_clayer(nn.Module):
    def __init__(self, k_size=3,pool_types=['avg','max']):
        super(eca_clayer, self).__init__()
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)


    def forward(self, x):
        channel_att_sum = 0
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x)
                channel_att_raw = self.conv(avg_pool.transpose(1, 2)).transpose(1,2)

            elif pool_type=='max':
                max_pool = self.maxpool(x)
                channel_att_raw = self.conv(max_pool.transpose(1, 2)).transpose(1, 2)

            channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum)
        return scale.expand_as(x)


class DeepCALayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepCALayer, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, ), nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, ), nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, ), nn.BatchNorm1d(hidden_dim))

        self.attn = eca_clayer(k_size=7)
    def forward(self, x):
        # Compute 1D convolutional part and apply global max pooling
        conv_out = self.dconv(x)
        x_c = conv_out * self.attn(conv_out)
        return F.relu(x_c, inplace=True)


class MVFF(nn.Module):

    def __init__(self,input_dims, num_classes,hidden_dim,device,n_view=3,r=4):
        super(MVFF, self).__init__()
        self.device = device
        self.n_view = n_view
        self.r = r
        self.sub_nets = nn.ModuleList([DeepCALayer(input_dim=input_dims[i],hidden_dim=hidden_dim)
                                   for i in range(n_view)])

        self.theta_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(r,1,kernel_size=1,padding=0,stride=1),nn.BatchNorm2d(1))
                                           for i in range(n_view)])

        self.rate1 = nn.Parameter(torch.Tensor(1))
        self.rate2 = nn.Parameter(torch.Tensor(1))
        self.softmax = nn.Softmax(1)

        self.ChannelGate = eca_clayer(k_size=7)

        self.concat_project = nn.Sequential(
            nn.Conv1d(hidden_dim * n_view,hidden_dim,kernel_size=3,padding=1,bias=False),
            nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        init_weights(self)


    def forward(self,data):
        # bert onehot pssm
        x = [data.x, data.x1, data.x2]

        all_out = []
        for id, net in enumerate(self.sub_nets):
            conv_attn = net(x[id])
            all_out.append(conv_attn)

        bert,onehot,pssm = all_out
        B,C,N = onehot.size()
        cov_matrix = cov(bert)
        onehot = onehot.matmul(cov_matrix) + onehot
        pssm = pssm.matmul(cov_matrix) + pssm

        temp_outs = [bert,onehot,pssm]
        theta_outs = []
        for id,layer in enumerate(self.theta_layers):
            theta = layer(temp_outs[id].view(B,self.r,-1,N)).view(B,-1, N)
            theta_outs.append(theta)

        theta_b,theta_o,theta_p = theta_outs
        alpha_bo = torch.matmul(theta_b.permute(0, 2, 1).contiguous(),theta_o) / N
        alpha_bp = torch.matmul(theta_b.permute(0, 2, 1).contiguous(),theta_p) / N
        score = self.softmax(alpha_bo + alpha_bp)

        onehot = onehot + self.rate1 * onehot.matmul(score)
        pssm = pssm + self.rate2 * pssm.matmul(score)

        scale = self.ChannelGate(bert)

        deep_outs = [bert * scale,onehot * scale,pssm * scale]
        deep_outs = torch.cat(deep_outs, dim=1)
        high_level_feat = self.concat_project(deep_outs)
        high_level_feat = self.maxpool(high_level_feat)
        high_level_feat = torch.flatten(high_level_feat, 1)

        y = self.fc_out(high_level_feat)
        y = torch.sigmoid(y)
        return y