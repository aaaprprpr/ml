#完整代码太多，把关键部分写在这里了，这是倒残差块的一个简单实现


#  Conv + BN + 激活函数
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation='relu'):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activation == 'relu' else h_swish(inplace=True)
        )

#  h-swish 激活函数
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x

#  SE 模块
class SE_Attention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SE_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return x * scale

#  倒残差块
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se, activation):
        super(InvertedResidual, self).__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1, activation=activation))
        layers.extend([
            ConvBNReLU(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels, activation=activation),
            SE_Attention(hidden_channels) if use_se else nn.Identity(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)
    
    
    
