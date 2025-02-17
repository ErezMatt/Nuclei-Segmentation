import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels//4),
            nn.ConvTranspose2d(in_channels//4, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
       
    def forward(self, x, skip = None):
        x = self.block(x)
        if skip is not None:
            x = x + skip  # Element-wise addition for residual connection
        return x

class LinkNet34(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.name = "LinkNet34"
        
        resnet = models.resnet34(pretrained=True)

        # Encoder path
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1  # 64 filters
        self.encoder3 = resnet.layer2  # 128 filters
        self.encoder4 = resnet.layer3  # 256 filters
        self.encoder5 = resnet.layer4  # 512 filters

        # Decoder path
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.conv_transpose = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.output = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # Initial conv layers
        e2 = self.encoder2(e1) # Residual Block 1
        e3 = self.encoder3(e2) # Residual Block 2
        e4 = self.encoder4(e3) # Residual Block 3
        e5 = self.encoder5(e4) # Residual Block 4 

        # Decoder
        d4 = self.decoder4(e5, e4)  # 512  => 256
        d3 = self.decoder3(d4, e3)  # 256 => 128
        d2 = self.decoder2(d3, e2)  # 128 => 64
        d1 = self.decoder1(d2)  # 64 => 64

        ct1 = self.conv_transpose(d1)
        c1 = self.conv1(ct1)
        c2 = self.conv2(c1)

        return self.output(c2)

if __name__ == '__main__':
    img = torch.rand((1, 3, 512, 512))
    linknet = LinkNet34(3, 10)
    linknet_out = linknet(img)
    assert linknet_out.size() == torch.Size([1, 10, 512, 512])
    print(f"U-Net Output Shape: {linknet_out.size()}")
   