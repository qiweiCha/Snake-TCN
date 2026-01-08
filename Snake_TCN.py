import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from lib.DSConv import DSConv
from torch.nn.utils import weight_norm
# 初始化参数
def InitWeights(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
        module.weight = nn.init.kaiming_normal_(module.weight, a=.02)
        if module.bias is not None:
            module.bias = nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class Gate(nn.Module):
    def __init__(self, in_channels):
       
        super(Gate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        
        g_processed = self.W_g(g)
        x_processed = self.W_x(x)

        
        if g_processed.shape != x_processed.shape:
             g_processed = F.interpolate(g_processed, size=x_processed.size()[2:], mode='bilinear', align_corners=True)

        
        combined = self.relu(g_processed + x_processed)
        
       
        alpha = self.psi(combined)

        output = x * alpha
        
        return output
    
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class MSCM(nn.Module):
    def __init__(self, in_channels, snake_kernel_sizes=[3, 3, 5, 5], device=None):
        super(MSCM, self).__init__()

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.num_branches = len(snake_kernel_sizes)

        self.branches = nn.ModuleList()
        for i in range(self.num_branches):
            k_size = snake_kernel_sizes[i]
            if k_size % 2 == 0 or k_size <= 1:
                raise ValueError(f"All snake_kernel_sizes must be odd integers > 1, but got {k_size}")
            
            morph_type = i % 2  
            self.branches.append(
                DSConv(in_ch=in_channels, 
                       out_ch=in_channels, 
                       kernel_size=k_size, 
                       extend_scope=1, 
                       morph=morph_type, 
                       if_offset=True, 
                       device=device)
            )

        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(self.num_branches)])
        self.gates = nn.ModuleList([Gate(in_channels) for _ in range(self.num_branches - 1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        branch_outputs = []
        for i in range(self.num_branches):
            out = self.branches[i](x)
            out = self.bn[i](out)
            branch_outputs.append(out)

        fused_feature = branch_outputs[0]
        for i in range(self.num_branches - 1):
            fused_feature = self.gates[i](fused_feature, branch_outputs[i+1])

        alpha = torch.sigmoid(self.gamma)
        ax = self.relu(alpha * fused_feature + (1 - alpha) * x)
        ax = self.conv_last(ax)

        return ax

class Res_conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_BN=True):
        super(Res_conv, self).__init__()
        
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1) if in_c != out_c else None
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp))
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.conv11 is not None:
            x = self.conv11(x)
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return out

class Conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_BN = True):
        super(Conv, self).__init__()
        
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1) if in_c != out_c else None
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
       

    def forward(self, x):
        if self.conv11 is not None:
            x = self.conv11(x)
        out = self.conv(x)
      
        return out

class Seq_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dp = 0,is_BN = True):
        super().__init__()
        self.SeqConv = Res_conv(in_channels, out_channels,dp = dp,is_BN = is_BN)
    def forward(self, x):
        sequence = []
        for i in range(x.shape[2]):
            image = self.SeqConv(x[:,:,i,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences

class Down(nn.Module):
    def __init__(self, in_c, out_c,dp,is_BN = True):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x

class Sequence_down(nn.Module):
    def __init__(self, in_c, out_c,dp,is_BN = True):
        super(Sequence_down, self).__init__()
        self.down = Down(in_c, out_c,dp,is_BN =is_BN)

    def forward(self, x):
        sequence = []
        for i in range(x.shape[1]):
            image = self.down(x[:,i,:,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_reshaped = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, C, S)
        output = self.network(x_reshaped)
        output_S = output.shape[2]
        output = output.view(B, H, W, -1, output_S).permute(0, 4, 3, 1, 2).contiguous() # B, S, C_out, H, W
        return output

class AttentionPooling(nn.Module):
    """
    注意力池化层
    """
    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_reshaped = x.view(B * S, C, H, W)
        
        score_reshaped = self.attention_conv(x_reshaped)
        
        scores = score_reshaped.view(B, S, 1, H, W)
        weights = F.softmax(scores, dim=1)

        context = (x * weights).sum(dim=1) # -> [B, C, H, W]
        
        return context

class ETEM(nn.Module):
    def __init__(self, input_channels, output_channels, is_down=True, dp=0.2, is_BN=True):
        super().__init__()
        self.input_channels = input_channels
        
        self.down = Sequence_down(input_channels * 2, output_channels, dp, is_BN=is_BN) if is_down else None
    
        tcn_channels = [output_channels, output_channels] 
        self.tcn = TemporalConvNet(
            num_inputs=output_channels,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dp
        )
        
        final_tcn_channels = tcn_channels[-1]
        self.attention_pooling = AttentionPooling(in_channels=final_tcn_channels)

    def forward(self, x, s=None):
        if self.down is not None:
            B, S, _, H, W = x.shape
            s = F.interpolate(s, size=(self.input_channels, H, W), mode='trilinear', align_corners=False)
            x = self.down(torch.cat([s, x], dim=2))
        sequences = self.tcn(x)
        last_sequence = self.attention_pooling(sequences)
        return sequences, last_sequence

class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c,is_BN = True):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm =nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c)
       

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out

class Up(nn.Module):
    def __init__(self, in_c, out_c, dp=0,is_BN = True):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),

            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False,is_BN = True):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c,is_BN = is_BN)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = Res_conv(out_c, out_c, dp,is_BN= is_BN )
        if self.is_up == True:
            self.up = Up(out_c, out_c//2, dp,is_BN= is_BN)
        if self.is_down == True:
            self.down = Down(out_c, out_c*2,dp,is_BN= is_BN)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class Snake_TCN(nn.Module):
    def __init__(self, input_reduce=[0,1,2,3,4,5,6,7], num_classes=2, num_channels=1, feature_scale=2,  dropout=0.2, fuse=True, out_ave=True):
        super(Snake_TCN, self).__init__()
        self.input_reduce = input_reduce
       
        self.num_channels = num_channels

        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]


        self.inc = Seq_conv(num_channels, filters[0],dp=dropout,is_BN=False)
        self.en1 = ETEM(filters[0],filters[0],is_down=False,is_BN=False)
        self.en2 = ETEM(filters[0],filters[1],is_BN=False)
        self.en3 = ETEM(filters[1],filters[2],is_BN=False)
        self.en4 = ETEM(filters[2],filters[3],is_BN=False)

        self.SAFM = MSCM(filters[2])

        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights)

    def forward(self, x):
        x=x.unsqueeze(2) 
        # 'b,t,c,h,w' -> 'b,1,t,c,h,w'
        if self.input_reduce == "mean":
            x = torch.mean(x, dim=2, keepdim=True)
        elif self.input_reduce == "min":
            x, _ = torch.min(x, dim=2, keepdim=True)
        elif isinstance(self.input_reduce, list):
            s = torch.split(x, 1, dim=1)
            seq = [s[i] for i in self.input_reduce]
            x = torch.cat(seq, dim=2)

        s = x.permute(0, 2, 1, 3, 4) 
        x = self.inc(x)
        x, sc1 = self.en1(x)
        x, sc2 = self.en2(x, s)
        x, sc3 = self.en3(x, s)
        _, sc4 = self.en4(x, s)

        x1_2, x_down1_2 = self.block1_2(sc1)

        x2_2, x_up2_2 = self.block2_2(sc2)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))

        x2_1, x_up2_1, x_down2_1 = self.block2_1(torch.cat([x_down1_2, x2_2], dim=1))

        x3_1, x_up3_1 = self.block3_1(sc3)

        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))

        x20, x_up20, x_down20 = self.block20(torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))

        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))

        _, x_up40 = self.block40(sc4)

        x_up40 = self.SAFM(x_up40) 

        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))

        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))

        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))

        x13 = self.block13(torch.cat([x12, x_up22], dim=1))

        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x10) +
                    self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        else:
            output = self.final5(x13)

        return output
    
