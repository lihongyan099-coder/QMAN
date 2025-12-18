import torch  
import torch.nn as nn  
from quaternion_layers import*

class REAL_PART(nn.Module):  
    def __init__(self, channel, k_size):  
        super(REAL_PART, self).__init__()  
        self.conv = nn.Conv2d(channel, 1, kernel_size=k_size, stride=1, padding=k_size // 2, bias=False)  
        self.relu = QuaternionReLU()

    def forward(self, x):  
        y = self.conv(x)  
        y = self.relu(y)  
        zero_imaginary = torch.zeros_like(y).expand(-1, 3, -1, -1)
        y = torch.cat([y, zero_imaginary], dim=1)  
        return y  


class IMAGINARY_PART(nn.Module):  
    def __init__(self, channel):  
        super(IMAGINARY_PART, self).__init__() 
        self.conv = nn.Conv1d(channel, 3, kernel_size=1, bias=False)  
        self.relu = QuaternionReLU()

    def forward(self, x):  
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = x.view(-1, c).unsqueeze(-1) 
        x = self.conv(x)  
        x = self.relu(x)  
        x = x.squeeze(-1)  
        x = x.view(b, h, w, 3).permute(0, 3, 1, 2).contiguous()  
        zero_real = torch.zeros_like(x[:, :1, :, :])  
        output = torch.cat([zero_real, x], dim=1)  
        return output 


class QuaternionLayer(nn.Module):  
    def __init__(self, channel, k_size):  
        super(QuaternionLayer, self).__init__()  
        self.real_part_layer = REAL_PART(channel, k_size)  
        self.imaginary_part_layer = IMAGINARY_PART(channel)  

    def forward(self, x):  
        real_part = self.real_part_layer(x)          # (B, 4, H', W')  
        imaginary_part = self.imaginary_part_layer(x) 
        quaternion_output = real_part + imaginary_part  
        return quaternion_output  
