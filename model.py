import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        """
        Squeeze-and-Excitation block with customizable reduction ratio.
        Defaults to 8 here instead of 16 for stronger channel attention.
        """
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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
        return x * w


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1,
                 use_se=False, se_reduction=8,
                 norm_layer=nn.BatchNorm2d):
        """
        Basic 3×3 + 3×3 ResNet block, optionally with Squeeze-and-Excitation.
        """
        super(BasicBlock, self).__init__()
        self.use_se = use_se

        # 1st conv
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        # 2nd conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)

        # Shortcut (identity or 1×1 conv)
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Identity()

        # SE block if requested
        self.se = SEBlock(out_channels * self.expansion, reduction=se_reduction) \
                  if use_se else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ModifiedResNet(nn.Module):
    def __init__(self,
                 num_blocks=(3, 5, 3),  # revised: 3 blocks, then 5, then 3
                 base_channels=64,
                 num_classes=10,
                 use_se=True,
                 se_reduction=8,      # stronger SE
                 norm_layer=nn.BatchNorm2d):
        """
        A ResNet-like architecture for CIFAR-10, with adjustable stages and SE.
        - num_blocks: tuple of blocks in each stage
        - base_channels: # of channels in first stage (will double each stage)
        - use_se: whether to add SE blocks
        - se_reduction: SE reduction ratio (8 is bigger attention than 16)
        - norm_layer: normalization layer (e.g. nn.BatchNorm2d)
        """
        super(ModifiedResNet, self).__init__()
        self.in_channels = base_channels
        self.use_se = use_se
        self.se_reduction = se_reduction

        # Initial 3×3 conv
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Build stages
        self.layers = nn.Sequential()
        channels = base_channels

        for i, blocks in enumerate(num_blocks):
            stride = 1 if i == 0 else 2  # downsample in stage > 0
            layer_blocks = []
            # First block in this stage (may downsample)
            layer_blocks.append(
                BasicBlock(self.in_channels, channels,
                           stride=stride, use_se=self.use_se,
                           se_reduction=self.se_reduction,
                           norm_layer=norm_layer)
            )
            self.in_channels = channels * BasicBlock.expansion

            # Remaining blocks in this stage
            for _ in range(1, blocks):
                layer_blocks.append(
                    BasicBlock(self.in_channels, channels,
                               stride=1, use_se=self.use_se,
                               se_reduction=self.se_reduction,
                               norm_layer=norm_layer)
                )

            self.layers.extend(layer_blocks)
            channels *= 2  # double channels for next stage

        # Global average pool + classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        # Initial conv + BN + ReLU
        out = self.relu(self.bn1(self.conv1(x)))
        # Stacked residual blocks
        out = self.layers(out)
        # Global pool + linear
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # Quick parameter-count check
    model = ModifiedResNet(num_blocks=(3, 5, 3),
                           base_channels=64,
                           num_classes=10,
                           use_se=True,
                           se_reduction=8)
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)

    num_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", out.shape)
    print("Total parameters:", num_params)
