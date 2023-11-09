import torch
from torch import nn, optim
from torch.nn import functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.relu = nn.SiLU(inplace=True)  # Using SiLU instead of ReLU
        self.conv1 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(middle_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class EfficientNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize decoder blocks, ensuring the channel sizes are compatible with the encoder
        self.decode_block1 = DecoderBlock(1280, 640, 320)
        self.decode_block2 = DecoderBlock(320, 160, 80)
        self.decode_block3 = DecoderBlock(80, 40, 20)
        self.decode_block4 = DecoderBlock(20, 10, 3)

        # The output convolution to get back to 3 channels for an RGB image
        self.output_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Assuming x is the output feature map from the encoder
        x = self.decode_block1(x)
        x = self.decode_block2(x)
        x = self.decode_block3(x)
        x = self.decode_block4(x)

        # The final output layer to get the right number of channels
        x = self.output_conv(x)
        # Ensure the output is between -1 and 1 for image data
        x = self.tanh(x)
        return x


# Example usage:
# Assuming `encoder_feature_map` is the output of the EfficientNet encoder with a feature map size of 8x8
decoder = EfficientNetDecoder()
reconstructed_image = decoder(encoder_feature_map)


