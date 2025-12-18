import torch  
from torch import nn
import math     
import numpy as np
import torch.nn.functional as F
import collections  
import scipy.io as sio
from quaternion_layers import*
from generatedq import*
from quaternion_layers import*
import torch.nn.init as init  
from util import*

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

in_channels = 4  
hidden_channels = [16,32,64, 128, 256] 

out_features = 14 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class DWA(nn.Module):

    def __init__(self, in_channels, out_channels,
        channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs
        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

 
class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            QuaternionLinear(channel, channel // reduction, bias=False),
            QuaternionReLU(),
            QuaternionLinear(channel // reduction, channel, bias=False),
            QuaternionSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    

class QrnnNet(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(QrnnNet, self).__init__()
        self.generatedq = QuaternionLayer(channel = 10, k_size = 3)  
        self.conv_1 =  QuaternionConv(4, hidden_channels[0], kernel_size=3, stride=1, padding=1) 
        self.conv_11 =  QuaternionConv(hidden_channels[0], hidden_channels[0], kernel_size=1, stride=1, padding=0)
        self.relu = QuaternionReLU()
        self.conv_2 = QuaternionConv(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=1, padding=1)
        self.conv_22 =  QuaternionConv(hidden_channels[1], hidden_channels[1], kernel_size=1, stride=1, padding=0)
        self.bn_1 = QuaternionBatchNorm2d(16, gamma_init=1.0, beta_param=True)
        self.bn_2 =  QuaternionBatchNorm2d(32, gamma_init=1.0, beta_param=True)
        self.pool_1 = nn.MaxPool2d(2, 2) 
        self.dropout_1 = nn.Dropout(0.25) 
        self.conv_3 = QuaternionConv(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=1)
        self.conv_33 =  QuaternionConv(hidden_channels[2], hidden_channels[2], kernel_size=1, stride=1, padding=0)
        self.conv_4 = QuaternionConv(hidden_channels[2], hidden_channels[3], kernel_size=3, stride=1, padding=1)
        self.conv_44 =  QuaternionConv(hidden_channels[3], hidden_channels[3], kernel_size=1, stride=1, padding=0)
        self.bn_3 =  QuaternionBatchNorm2d(64, gamma_init=1.0, beta_param=True)
        self.bn_4 =  QuaternionBatchNorm2d(128, gamma_init=1.0, beta_param=True)
        self.pool_2 = nn.MaxPool2d(1, 1)
        self.dropout_2 = nn.Dropout(0.25)
        self.fc1 = {}  
        self.fc_2 = QuaternionLinear(512, out_features)  
        self.dropout_3 = nn.Dropout(0.5)  
        self.sm = nn.Softmax(dim=1)       
        
        self.se1 = SELayer(hidden_channels[0])
        self.se2 = SELayer(hidden_channels[1])
        self.se3 = SELayer(hidden_channels[2])
        self.se4 = SELayer(hidden_channels[3])
        
        for m in self.modules():    
            if isinstance(m, nn.Conv2d):   
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) 
                m.bias.data.zero_()   
        

    def forward(self, x):
      x = self.generatedq(x)
      y = self.conv_1(x)
      y = self.pool_1(y)
      y = self.bn_1(y)  
      y= self.relu(y)
      y = self.se1(y)
      y= self.conv_11(y)
      y = self.pool_2(y)
      y = self.bn_1(y)  
      y = self.relu(y)  

      y = torch.flatten(y, start_dim=1) 
      in_features = y.shape[1]  
      if in_features not in self.fc1:  
        self.fc1[in_features] = QuaternionLinear(in_features, 512).to(device)  
      y = y.to(device)
      y = self.fc1[in_features](y)
      y = y.to(device)
      y = self.relu(y)
      y = self.dropout_3(y)     
      y = self.fc_2(y)  
      y = self.sm(y)    
      return y



class MultiBranchNet(nn.Module):  
    def __init__(self, input_channels, flatten=True, num_branches=3, num_sub_branches=3, out_features=14):  
        super(MultiBranchNet, self).__init__()  
        self.num_branches = num_branches
        self.num_sub_branches = num_sub_branches
        self.name = "QMAN"  
        self.branches = nn.ModuleList([
            nn.ModuleList([QrnnNet(input_channels=input_channels, flatten=flatten) for _ in range(num_sub_branches)])
            for _ in range(num_branches)
        ])
        self.fc = nn.Linear(num_branches * num_sub_branches * 6*2 , out_features)  
        self.init_weights()  
        self.atten = DWA(in_channels=30, out_channels=14, channelAttention_reduce=4)
        
        self.patch_sizes = [17,19,21]  



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def extract_patch(self, x, patch_size):
        height, width = x.shape[1], x.shape[2]
        half_size = patch_size // 2
        if patch_size == height:
            return x 
        center_h, center_w = height // 2, width // 2
        start_h = center_h - half_size
        end_h = center_h + half_size + 1
        start_w = center_w - half_size
        end_w = center_w + half_size + 1
        return x[:, start_h:end_h, start_w:end_w, :]


    def forward(self, x): 
        x = x.permute(0, -1, 2, 1)
        x = self.atten(x)  
        x = x.permute(0, -1, 2, 1)
        batch_size, height, width, spectral_bands = x.shape
        splits = [10,20,30]  
        
        branch_1_input = x[:, :, :, :splits[0]]  
        branch_2_input = x[:, :, :, :splits[0]] + x[:, :, :, splits[0]:splits[1]]  
        branch_3_input = x[:, :, :, :splits[0]] + x[:, :, :, splits[0]:splits[1]] + x[:, :, :, splits[1]:splits[2]] 
        branch_inputs = [branch_1_input, branch_2_input, branch_3_input]
        all_branch_outputs = []

        for branch_idx, branch in enumerate(self.branches):  
            branch_input = branch_inputs[branch_idx]  
            sub_branch_outputs = []

            for sub_branch_idx, sub_branch in enumerate(branch):  
                patch_size = self.patch_sizes[sub_branch_idx]
                patch_input = self.extract_patch(branch_input, patch_size)
                patch_input = patch_input.permute(0, 3, 1, 2) 
                sub_branch_output = sub_branch(patch_input)  
                sub_branch_outputs.append(sub_branch_output)
            if sub_branch_outputs:
                sub_branch_outputs = torch.stack(sub_branch_outputs, dim=1)  # [batch_size, num_sub_branches, 4]
                all_branch_outputs.append(sub_branch_outputs)
            else:
                raise ValueError(f"No outputs generated for branch {branch_idx}. Check input data or patch extraction logic.")

        if all_branch_outputs:
            all_branch_outputs = torch.cat(all_branch_outputs, dim=1) 
            output = self.fc(all_branch_outputs.view(all_branch_outputs.size(0), -1))
            output = torch.softmax(output, dim=1)
        else:
            raise ValueError("No branch outputs generated; check your input data.")
        return output
