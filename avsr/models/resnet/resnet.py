import torch
import torch.nn as nn


def get_residual_layer(
        num_layer, in_channels, out_channels, kernel_size, dim=1, identity=False):
    layers = nn.Sequential()
    for i in range(num_layer):
        if i == 0:
            params = [in_channels, out_channels, kernel_size, identity]
        else:
            params = [out_channels, out_channels, kernel_size] 
        layer = ResidualCell(*params) if dim==1 else ResidualCell2d(*params)
        layers.add_module(f'{i}', layer)
    return layers

    
class ResidualCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, identity=True):
        super().__init__()
        if identity:
            self.shortcut = nn.Identity(stride=2)
            stride = 1
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels,out_channels,kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())

    def forward(self, x):
        Fx = self.conv(x)
        x = self.shortcut(x)
        return Fx + x
    

class ResidualCell2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, identity=True):
        super().__init__()
        if identity:
            self.shortcut = nn.Identity(stride=2)
            stride = 1
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            stride = 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    
    def forward(self, x):
        Fx = self.conv(x)
        x = self.shortcut(x)
        return Fx + x


class Resnet1D_front(nn.Module):
    '''
    input :  (BxCxL)
    output : (BxDxL`) L`:= length of audio sequence with 30 Hz
    '''
    def __init__(self, n_channels : int = 2, out_dim : int = 512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, 
                      kernel_size=79, stride=3, 
                      padding=39), # 80 : 5ms
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.conv2 = get_residual_layer(2, 64, 64, 3, identity=True)
        self.conv3 = get_residual_layer(2, 64, 128, 3)
        self.conv4 = get_residual_layer(2, 128, 256, 3)
        self.conv5 = get_residual_layer(2, 256, out_dim, 3)
        self.avg_pool = nn.AvgPool1d(21, 20, padding=10) # -> 30fps
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.avg_pool(outputs)
        return outputs
        
        
class Resnet2D_front(nn.Module):
    '''
    input :  (BxCxLxHxW)
    output : (BxLxD)
    '''
    def __init__(self, n_channels : int = 3, out_dim : int = 512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_channels, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,3,3),(1,2,2)))
        self.conv2 = get_residual_layer(2, 64, 64, 3, dim=2, identity=True)
        self.conv3 = get_residual_layer(2, 64, 128, 3, dim=2)
        self.conv4 = get_residual_layer(2, 128, 256, 3, dim=2)
        self.conv5 = get_residual_layer(2, 256, out_dim, 3, dim=2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, inputs):
        outputs = self.conv1(inputs).permute(0,2,1,3,4)
        outputs = outputs.flatten(end_dim=1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.avg_pool(outputs)
        outputs = outputs.view(inputs.shape[0], inputs.shape[2], -1)
        return outputs
