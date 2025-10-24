import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out


class CNNBase(nn.Module):
    def __init__(self,
                 pooling_type=False,
                 is_resnet=False,
                 is_large_hidden=False,
                 is_large_layer=False,
                 label_num=10,
                 input_size=32):
        super(CNNBase, self).__init__()
        self.pooling_type = 'max' if not pooling_type else 'mean'
        self.is_resnet = is_resnet
        self.is_large_hidden = is_large_hidden
        self.is_large_layer = is_large_layer
        self.input_size = input_size
        
        multiplier = 2 if is_large_hidden else 1
        
        if is_large_layer:
            self.channels = [3, 128*multiplier, 256*multiplier, 512*multiplier,
                           512*multiplier, 512*multiplier, 512*multiplier, 256*multiplier]
            self.num_layers = 7
            self.pool_after = [0, 1, 6]
        else:
            self.channels = [3, 128*multiplier, 256*multiplier, 512*multiplier,
                           512*multiplier, 256*multiplier]
            self.num_layers = 5
            self.pool_after = [0, 1, 4]
        
        self.conv_layers = nn.ModuleList()
        if is_resnet:
            for i in range(self.num_layers):
                self.conv_layers.append(
                    ResidualBlock(self.channels[i], self.channels[i + 1], stride=1)
                )
        else:
            for i in range(self.num_layers):
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(self.channels[i], self.channels[i + 1],
                             kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ))
        
        if self.pooling_type == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        elif self.pooling_type == 'mean':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        self.conv_dropout = nn.Dropout(0.3)
        
        num_pools = len(self.pool_after)
        feature_size = input_size // (2 ** num_pools)
        fc_input_dim = self.channels[-1] * feature_size * feature_size
        
        fc_hidden1 = 512 * multiplier
        fc_hidden2 = 256 * multiplier
        fc_hidden3 = 128 * multiplier
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden2, fc_hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden3, label_num)
        )
    
    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i in self.pool_after:
                x = self.pool(x)
                x = self.conv_dropout(x)
        x = self.fc_layers(x)
        return x
    def get_config_str(self):
        return (f"layers{self.base_layer_num}_"
                f"dim{self.base_hidden_dim}x_"
                f"{self.pooling_type}pool_"
                f"{'resnet' if self.is_resnet else 'plain'}")

