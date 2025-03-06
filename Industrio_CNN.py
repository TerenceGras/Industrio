import torch.nn as nn


# ---------------------------
# Define the CNN architecture
# ---------------------------
class iCNN(nn.Module):
    def __init__(self):
        super(iCNN, self).__init__()

        # Block 1: Basic low-level feature extraction.
        # Input: Grayscale image (1 channel), 416x416.
        # Output: 32 channels, 416x416, then downsampled to 208x208.
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # 416x416 -> 416x416
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 416x416 -> 208x208
        )

        # Block 2: Mid-level feature extraction.
        # Input: 32 channels, 208x208.
        # Output: 64 channels, 208x208, then downsampled to 104x104.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 208x208 -> 208x208
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 208x208 -> 104x104
        )

        # Block 3: Feature refinement using a bottleneck structure.
        # Input: 64 channels, 104x104.
        # Expand to 128, compress to 64 via 1x1, expand again to 128, then downsample to 52x52.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 104x104, 128 channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),  # Bottleneck: 104x104, 64 channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 104x104, 128 channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 104x104 -> 52x52
        )

        # Block 4: Higher-level feature extraction for finer details.
        # Input: 128 channels, 52x52.
        # Downsample to 26x26.
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 52x52, 256 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),  # Bottleneck: 52x52, 128 channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 52x52, 256 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 52x52 -> 26x26
        )

        # Block 5: Optional additional refinement for very fine details.
        # Input: 256 channels, 26x26.
        # Downsample to 13x13.
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 26x26, 512 channels
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),  # Bottleneck: 26x26, 256 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 26x26, 512 channels
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 26x26 -> 13x13
        )

        # Global average pooling to convert final feature maps into a fixed-size vector.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch, channels, 1, 1)

        # Fully connected layer to produce a single scalar output.
        # Using 512 channels from block5 (flattened to 512-dim).
        self.fc = nn.Linear(512, 1)

        # Sigmoid activation to map the output to a confidence score in [0, 1].
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)  # (batch, 32, 208, 208)
        x = self.block2(x)  # (batch, 64, 104, 104)
        x = self.block3(x)  # (batch, 128, 52, 52)
        x = self.block4(x)  # (batch, 256, 26, 26)
        x = self.block5(x)  # (batch, 512, 13, 13)
        x = self.global_pool(x)  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 512)
        x = self.fc(x)  # (batch, 1)
        confidence = self.sigmoid(x)  # Output confidence score between 0 and 1.
        return confidence
