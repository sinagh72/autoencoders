import torch
from torch import nn, optim
from torch.nn import functional as F
class Autoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 256,
                 height: int = 256):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer_type == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd, momentum=self.hparams.momentum)
        elif self.hparams.optimizer_type == 'ADAM':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer_type == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer_type == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
            scheduler = None  # Adagrad typically does not require a scheduler
        elif self.hparams.optimizer_type == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
            scheduler = None  # Adadelta typically does not require a scheduler
        elif self.hparams.optimizer_type == 'Nadam':
            optimizer = optim.NAdam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        else:
            raise ValueError(f"Unknown optimizer type: {self.hparams.optimizer_type}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Assuming you want to monitor validation loss to adjust the LR
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
