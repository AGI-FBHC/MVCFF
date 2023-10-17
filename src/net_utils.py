import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, k_size=3,pool_types=['avg','max']):
        super(ChannelGate, self).__init__()
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)


    def forward(self, x):
        channel_att_sum = 0
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x).permute(0,2,1).contiguous()
                channel_att_raw = self.conv(avg_pool).permute(0,2,1).contiguous()

            elif pool_type=='max':
                max_pool = self.maxpool(x).permute(0,2,1).contiguous()
                channel_att_raw = self.conv(max_pool).permute(0,2,1).contiguous()

            channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).expand_as(x)
        return scale


class DeepCALayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepCALayer, self).__init__()

        self.base_conv = nn.Sequential(
            nn.Conv1d(input_dim,hidden_dim,kernel_size=3,padding=1),nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim,hidden_dim,kernel_size=3,padding=1),nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim,hidden_dim,kernel_size=3,padding=1),nn.BatchNorm1d(hidden_dim)
        )
        self.ChannelGate = ChannelGate(k_size=7)

    def forward(self, x):
        base_out = self.base_conv(x)
        c_weight = self.ChannelGate(base_out)
        c_out = base_out * c_weight
        return F.relu(c_out)


class SeparableConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv, self).__init__()
        self.conv1 =nn.Sequential(nn.Conv1d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias),nn.BatchNorm1d(inplanes))
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x.unsqueeze(2))
        return x.squeeze(2)


def conv_2(in_planes,out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False), nn.BatchNorm2d(out_planes),nn.ReLU(inplace=True))


def cov(input):
    b,c,h = input.size()
    x = input - torch.mean(input)
    x = x.view(b*c,h)
    cov_matrix = torch.matmul(x.T,x) / x.shape[0]
    return cov_matrix

# class ChannelPool(nn.Module):
#
#     def forward(self, x):
#         # return torch.cat((torch.max(x,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)
#         return torch.mean(x,1).unsqueeze(1)
class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2,1,kernel_size=1,padding=0,bias=False),
            nn.ReLU()
        )
    def forward(self, x):
        global_x = torch.cat((torch.max(x,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)
        global_x = self.conv(global_x)
        return global_x