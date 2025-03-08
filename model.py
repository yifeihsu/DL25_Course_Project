import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Two fully-connected layers (implemented as 1x1 conv for convenience)
        self.fc1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w  # scale the input features by the learned weights

# Basic Residual Block (for CIFAR-10 ResNet) with optional SE attention
class BasicBlock(nn.Module):
    expansion = 1  # expansion factor for output channels (1 for BasicBlock)
    def __init__(self, in_channels, out_channels, stride=1, use_se=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.use_se = use_se
        # First 3x3 convolution (with stride, to downsample if needed)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        # Shortcut (identity or 1x1 conv if shape mismatch)
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            # Use 1x1 conv to match dimensions (for downsampling or channel increase)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1,
                          stride=stride, bias=False),
                norm_layer(out_channels * BasicBlock.expansion)
            )
        else:
            self.shortcut = nn.Identity()
        # Squeeze-and-Excitation module (if use_se is True)
        self.se = SEBlock(out_channels * BasicBlock.expansion) if use_se else nn.Identity()
        # Activation (ReLU) - we will apply after each BN, and after adding shortcut
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                 # SE attention (if no SE, this is identity)
        out += self.shortcut(x)            # add skip connection
        out = self.relu(out)
        return out

# Modified ResNet model for CIFAR-10
class ModifiedResNet(nn.Module):
    def __init__(self, num_blocks=[4,4,3], base_channels=64, num_classes=10, use_se=True, norm_layer=nn.BatchNorm2d):
        """
        num_blocks: list of number of residual blocks in each stage.
        base_channels: number of channels in first stage (will double each stage).
        use_se: whether to include SE attention in blocks.
        norm_layer: normalization layer to use (default BatchNorm2d).
        """
        super(ModifiedResNet, self).__init__()
        self.in_channels = base_channels
        self.use_se = use_se
        # Initial conv layer (3x3, same padding)
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1  = norm_layer(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # Build residual stages
        self.layers = nn.Sequential()  # will hold all residual blocks
        channels = base_channels
        for i, blocks in enumerate(num_blocks):
            # For each stage i:
            # Set stride=2 for stages beyond the first to downsample spatially.
            stride = 1 if i == 0 else 2
            # First block of this stage (may downsample and/or increase channels)
            layer_blocks = []
            layer_blocks.append(BasicBlock(self.in_channels, channels, stride=stride,
                                           use_se=self.use_se, norm_layer=norm_layer))
            self.in_channels = channels * BasicBlock.expansion  # update current channels
            # Remaining blocks of this stage (stride=1, same channel count)
            for j in range(1, blocks):
                layer_blocks.append(BasicBlock(self.in_channels, channels, stride=1,
                                               use_se=self.use_se, norm_layer=norm_layer))
                # in_channels remains the same within a stage for BasicBlock
            # Append this stage's blocks to the Sequential
            self.layers.extend(layer_blocks)
            # Double the channel count for next stage
            channels *= 2
        # Global average pool and linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        # Initial conv + BN + ReLU
        out = self.relu(self.bn1(self.conv1(x)))
        # Residual blocks
        out = self.layers(out)
        # Global pooling and output
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)   # flatten
        out = self.fc(out)
        return out