import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):

        super(DepthwiseSeparableConv, self).__init__()
        
                        
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
                           
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
              
        x = self.depthwise(x)
        x = self.bn1(x)
        
              
        x = self.pointwise(x)
        x = self.bn2(x)
        
        return x


class DepthwiseSeparableConvWithReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConvWithReLU, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class DepthwiseSeparableConvSequential(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_relu=True, bias=False):
        super(DepthwiseSeparableConvSequential, self).__init__()
        
        if use_relu:
            self.conv = DepthwiseSeparableConvWithReLU(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )
        else:
            self.conv = DepthwiseSeparableConv(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )
    
    def forward(self, x):
        return self.conv(x)


def replace_conv_with_depthwise(module, in_channels, out_channels, kernel_size=3, 
                                 stride=1, padding=1, use_relu=False):

    if use_relu:
        return DepthwiseSeparableConvWithReLU(
            in_channels, out_channels, kernel_size, stride, padding
        )
    else:
        return DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size, stride, padding
        )


         
def calculate_conv_params(in_channels, out_channels, kernel_size):
                    
    return in_channels * out_channels * kernel_size * kernel_size


def calculate_depthwise_params(in_channels, out_channels, kernel_size):
                       
    return in_channels * kernel_size * kernel_size + in_channels * out_channels


def calculate_reduction_ratio(in_channels, out_channels, kernel_size):
                  
    standard = calculate_conv_params(in_channels, out_channels, kernel_size)
    depthwise = calculate_depthwise_params(in_channels, out_channels, kernel_size)
    return (standard - depthwise) / standard * 100
