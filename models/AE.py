import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl

from models.ssim import SSIMLoss


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 encoder,
                 decoder,
                 loss_type,
                 device,
                 **kwargs):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.loss_type = loss_type
        self.kwargs = kwargs

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
        if self.loss_type == "mse":
            loss = F.mse_loss(x, x_hat)
        elif self.loss_type == "cosine":
            x_flat = x.view(self.kwargs["batch_size"], -1)
            x_hat_flat = x_hat.view(self.kwargs["batch_size"], -1)
            loss = 1 - F.cosine_similarity(x_flat, x_hat_flat, dim=1).mean()
        elif self.loss_type == "ssim":
            loss = SSIMLoss()
        return loss

    def configure_optimizers(self):
        if self.kwargs["optimizer_type"] == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"])
        elif self.kwargs["optimizer_type"] == 'SGD_nesterov':
            optimizer = optim.SGD(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                  nesterov=True, momentum=0.9)
        elif self.kwargs["optimizer_type"] == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                   betas=(0.9, 0.999), eps=1e-8)
        elif self.kwargs["optimizer_type"] == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                    betas=(0.9, 0.999), eps=1e-8)
        elif self.kwargs["optimizer_type"] == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                      eps=1e-8)
        elif self.kwargs["optimizer_type"] == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                      eps=1e-8)
        elif self.kwargs["optimizer_type"] == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                       eps=1e-8)
        elif self.kwargs["optimizer_type"] == 'Nadam':
            optimizer = optim.NAdam(self.parameters(), lr=self.kwargs["lr"], weight_decay=self.kwargs["wd"],
                                    eps=1e-8)
        else:
            raise ValueError(f"Unknown optimizer type: {self.kwargs['optimizer_type']}")

        if self.kwargs["scheduler_type"] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.kwargs["step_size"], gamma=0.5)

        elif self.kwargs["scheduler_type"] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)

        elif self.kwargs["scheduler_type"] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                             threshold=0.0001, cooldown=3, min_lr=1e-8)
        else:
            raise ValueError(f"Unknown scheduler type: {self.kwargs['scheduler']}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Assuming you want to monitor validation loss to adjust the LR
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss, on_step=True, sync_dist=True)


class MaskedAutoencoder(Autoencoder):
    def __init__(self, encoder, decoder, loss_type, device, **kwargs):
        super().__init__(encoder, decoder, loss_type, device, **kwargs)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        img, mask = x
        z, m_hat = self.encoder(x, mask)
        x_hat = self.decoder(z)
        return x_hat, m_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        (img, mask), _ = batch  # We do not need the labels
        img_hat, m_hat = self.forward((img, mask))
        if self.loss_type == "l1":
            loss_recon = F.l1_loss(img, img_hat, reduction='none')
            # 3 is input channels
            loss = (loss_recon * m_hat).sum() / (m_hat.sum() + 1e-8) / 3
        elif self.loss_type == "cosine":
            x_flat = img.view(self.kwargs["batch_size"], -1)
            x_hat_flat = img_hat.view(self.kwargs["batch_size"], -1)
            loss = 1 - F.cosine_similarity(x_flat, x_hat_flat, dim=1).mean()
        return loss
