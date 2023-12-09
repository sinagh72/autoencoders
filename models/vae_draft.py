# pip install pytorch-lightning
# pip install efficientnet_pytorch
# pip install torch torchvision

#import packages
import torch
from torch import nn
import timm
# from efficientnet_pytorch import EfficientNet


#initializing code


class DecoderBlock(nn.Module):
    # ... [Keep the DecoderBlock unchanged] ...

    def __init__(self, in_channels, middle_channels, out_channels, upsample_stride=2, output_padding=1):
        super().__init__()
        # Adjust the stride of the transposed convolution to achieve the desired upscaling
        self.upconv = nn.ConvTranspose2d(
            in_channels, middle_channels, kernel_size=3, stride=upsample_stride, padding=1,
            output_padding=output_padding
        )
        self.relu = nn.SiLU(inplace=True)  # Using SiLU instead of ReLU
        self.conv1 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(middle_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        # print("type of x before upconv", type(x))
        # print("shape of x before upconv", x.shape if isinstance(x,torch.Tensor) else "Not a tensor")
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
    # ... [Keep the EfficientNetDecoder unchanged] ...

    def __init__(self):
        super().__init__()
        # Initialize decoder blocks with strides and paddings adjusted to increase the size without explicit upsampling
        self.decode_block1 = DecoderBlock(1280, 640, 320, upsample_stride=2, output_padding=1)  # 8x8 to 16x16
        self.decode_block2 = DecoderBlock(320, 160, 80, upsample_stride=2, output_padding=1)  # 16x16 to 32x32
        self.decode_block3 = DecoderBlock(80, 40, 20, upsample_stride=2, output_padding=1)  # 32x32 to 64x64
        self.decode_block4 = DecoderBlock(20, 10, 3, upsample_stride=4, output_padding=3)  # 64x64 to 256x256

        # The output convolution to get back to 3 channels for an RGB image
        self.output_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print("type of x begins", type(x))
        # print("shape of x begins", x.shape if isinstance(x,torch.Tensor) else "Not a tensor")
        x = self.decode_block1(x)
        # print("testing decode block1", x.shape)
        x = self.decode_block2(x)
        # print("testing decode block2", x.shape)
        x = self.decode_block3(x)
        # print("testing decode block3", x.shape)
        x = self.decode_block4(x)
        # print("testing decode block4", x.shape)


        # The final output layer to get the right number of channels
        x = self.output_conv(x)

        # print("type of x after output_conv", type(x))
        # print("shape of x output_conv", x.shape if isinstance(x,torch.Tensor) else "Not a tensor")
        # Ensure the output is between -1 and 1 for image config
        x = self.tanh(x)
        # print ("type of x after tanh is ", type(x))
        # print ("shape of x after tanh", x.shape if isinstance(x,torch.tensor) else "not a tensor")
        return x

def get_EfficientNetEncoder():
    # ... [Keep the EfficientNet encoder extraction unchanged] ...

    model = timm.create_model(
        'tf_efficientnetv2_s.in21k_ft_in1k',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
        drop_rate=0.1,
        drop_path_rate=0.1,
    )
    model.global_pool = nn.Identity()    
    return model

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = get_EfficientNetEncoder()
        self.fc_mu = nn.Conv2d (1280, 1280, kernel_size =1) # Assuming 1280 is the feature size of the encoder
        self.fc_logvar = nn.Conv2d (1280, 1280, kernel_size =1)
        self.decoder = EfficientNetDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)
        # print("after encoder feature size is", features.shape)
        #flatten the features to match expected self.fc_mu dim
        # features = features.view(features.size(0), -1)

        # print("later, feature size is ",features.shape)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        # z = z.unsqueeze(-1).unsqueeze(-1)  # Reshape z to match the expected input shape of the decoder

        # print("type of z", type(z))
        # print("shape of z is", z.shape if isinstance(z,torch.Tensor) else "Not a tensor")

        return z
        # return z, mu, logvar

# if __name__ == "__main__":
#     latent_dim = 128  # Define the size of the latent space, CHECK IF NEEDED
#     tensor = torch.rand((1, 3, 256, 256))
#
#     decoder = EfficientNetDecoder()
#     vae = VariationalAutoencoder(latent_dim=latent_dim)
#     reconstructed_vae, mu, logvar = vae(tensor)
#     print(reconstructed_vae.shape, mu.shape, logvar.shape)
#
#     out = decoder(out)
#     print(out.shape)

