import os
import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from DataModule import KermanyDataModule
from models.AE import Autoencoder
from models.vae_draft import VariationalAutoencoder, EfficientNetDecoder
from utils.transformations import get_train_transformation
from utils.util import set_seed
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers

if __name__ == "__main__":
    max_epochs = 200
    set_seed(10)

    kermany_classes = [("NORMAL", 0),
                       ("DRUSEN", 1),
                       ("DME", 2),
                       ("CNV", 3),
                       ]
    img_size = 256
    batch_size = 72
    load_dotenv(dotenv_path="./config/.env")
    dataset_path = os.getenv('KERMANY')
    kermany_datamodule = KermanyDataModule(data_dir=dataset_path,
                                           batch_size=batch_size,
                                           classes=kermany_classes,
                                           split=[0.95, 0.05],
                                           train_transform=get_train_transformation(size=img_size),
                                           num_workers=torch.cuda.device_count() * 2
                                           )
    # preparing config
    kermany_datamodule.prepare_data()
    kermany_datamodule.setup("train")
    kermany_datamodule.setup("val")
    train_loader = kermany_datamodule.train_dataloader()
    val_loader = kermany_datamodule.val_dataloader()
    print(len(train_loader))

    loss_types = "mse", "cosine"
    optimizer_types = ["SGD", "Nesterov", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Nadam"]
    scheduler_types = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    patience = 10
    step_size = len(train_loader) // batch_size * patience
    lr = 1e-4
    wd = 1e-6
    for scheduler_type in scheduler_types:
        for optimizer_type in optimizer_types:
            for loss_type in loss_types:
                model_path = "vae_checkpoints/vae_" + loss_type + "_" + optimizer_type + "_" + scheduler_type
                if os.path.exists(model_path):
                    continue

                # Model initialization
                model = Autoencoder(encoder=VariationalAutoencoder(1280),
                                    decoder=EfficientNetDecoder(),
                                    loss_type=loss_type,
                                    optimizer_type=optimizer_type,
                                    scheduler_type=scheduler_type,
                                    step_size=step_size,
                                    lr=lr,
                                    wd=wd,
                                    device="cuda:0",
                                    batch_size=batch_size)
                # Set up logging for training progress
                csv_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "csv_log/"))
                tb_logger = TensorBoardLogger(save_dir=os.path.join(model_path, "tb_log/"), name="vae")
                # Define early stopping criteria
                monitor = "val_loss"
                mode = "min"
                early_stopping = EarlyStopping(monitor=monitor, patience=patience, verbose=False, mode=mode)
                # Initialize the trainer and start training
                trainer = pl.Trainer(
                    strategy="ddp_find_unused_parameters_true",
                    default_root_dir=model_path,
                    accelerator="gpu",
                    log_every_n_steps=1,
                    devices=4,
                    max_epochs=max_epochs,
                    callbacks=[
                        early_stopping,
                        ModelCheckpoint(dirpath=model_path, filename="vae-{epoch}-{val_loss:.2f}", save_top_k=1,
                                        save_weights_only=True, mode=mode, monitor=monitor),
                    ],
                    logger=[tb_logger, csv_logger],
                )
                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
