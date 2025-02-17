import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)
        # If input and output channels are different, use 1x1 conv to match dimensions
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x) # Features passed to the decoder (skip connection)
        down = self.pool(skip) # Downsampled feature map

        return down, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], 1) # Concatenating with skip connection
        return self.conv(x)

class UNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block):
        super().__init__()
        self.name = "ResUNet" if conv_block == ResidualBlock else "UNet"
        
        # Encoder path
        self.encoder_block1 = EncoderBlock(in_channels, 64, conv_block)
        self.encoder_block2 = EncoderBlock(64, 128, conv_block)
        self.encoder_block3 = EncoderBlock(128, 256, conv_block)
        self.encoder_block4 = EncoderBlock(256, 512, conv_block)

        # Bottleneck
        self.down = conv_block(512, 1024)

        # Decoder path
        self.decoder_block1 = DecoderBlock(1024, 512, conv_block)
        self.decoder_block2 = DecoderBlock(512, 256, conv_block)
        self.decoder_block3 = DecoderBlock(256, 128, conv_block)
        self.decoder_block4 = DecoderBlock(128, 64, conv_block)

        # Final output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1, s1 = self.encoder_block1(x)
        e2, s2 = self.encoder_block2(e1)
        e3, s3 = self.encoder_block3(e2)
        e4, s4 = self.encoder_block4(e3)

        # Bottleneck
        d = self.down(e4)

        # Decoder
        d1 = self.decoder_block1(d, s4)
        d2 = self.decoder_block2(d1, s3)
        d3 = self.decoder_block3(d2, s2)
        d4 = self.decoder_block4(d3, s1)

        out = self.output(d4)
        return out

class UNet(UNetBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, DoubleConv)

class ResUNet(UNetBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, ResidualBlock)

if __name__ == '__main__':
    img = torch.rand((1, 3, 512, 512))
    unet = UNet(3, 10)
    unet_out = unet(img)
    assert unet_out.size() == torch.Size([1, 10, 512, 512])
    print(f"U-Net Output Shape: {unet_out.size()}")
    resunet = ResUNet(3, 10)
    resunet_out = resunet(img)
    assert resunet_out.size() == torch.Size([1, 10, 512, 512])
    print(f"ResU-Net Output Shape: {resunet_out.size()}")
