import copy
import os
import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from DataModule import SrinivasanDataModule, OCT500DataModule
from models.AE import Autoencoder
from models.base import ClsModel
from models.vae_draft import VariationalAutoencoder, EfficientNetDecoder
from utils.transformations import get_train_transformation, get_test_transformation
from utils.util import set_seed
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers

if __name__ == "__main__":
    max_epochs = 10
    set_seed(10)



    img_size = 256
    batch_size = 72
    load_dotenv(dotenv_path="./config/.env")
    """Srinivasan"""
    srinivasan_classes = [("NORMAL", 0),
                          ("AMD", 1),
                          ("DME", 2),
                          ]
    srinivasan_dataset_path = os.getenv('SRINIVASAN')
    srinivasan_datamodule = SrinivasanDataModule(data_dir=srinivasan_dataset_path,
                                                 batch_size=batch_size,
                                                 classes=srinivasan_classes,
                                                 split=[0.8, 0, 0.2],
                                                 train_transform=get_train_transformation(size=img_size),
                                                 test_transform=get_test_transformation(size=img_size),
                                                 num_workers=torch.cuda.device_count() * 2
                                                 )
    # preparing config
    srinivasan_datamodule.prepare_data()
    srinivasan_datamodule.setup("train")
    srinivasan_datamodule.setup("test")
    srinivasan_train_loader = srinivasan_datamodule.train_dataloader()

    """oct500"""
    oct500_dataset_path = os.getenv('OCT500')
    oct500_classes = [("NORMAL", 0),
                      ("AMD", 1),
                      ('DR', 2),
                      ('CNV', 3),
                      ('OTHERS', 4),
                      ]
    oct500_datamodule = OCT500DataModule(data_dir=oct500_dataset_path,
                                         batch_size=batch_size,
                                         classes=oct500_classes,
                                         split=[0.8, 0, 0.2],
                                         train_transform=get_train_transformation(size=img_size),
                                         test_transform=get_test_transformation(size=img_size),
                                         num_workers=torch.cuda.device_count() * 2
                                         )
    # preparing config
    oct500_datamodule.prepare_data()
    oct500_datamodule.setup("train")
    oct500_datamodule.setup("test")
    oct500_train_loader = oct500_datamodule.train_dataloader()

    loss_types = "mse", "cosine"
    optimizer_types = ["SGD", "SGD_nesterov", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Nadam"]
    scheduler_types = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    patience = 3
    srinivasan_step_size = len(srinivasan_train_loader) // batch_size * patience
    lr = 1e-4
    wd = 1e-6
    for scheduler_type in scheduler_types:
        for optimizer_type in optimizer_types:
            for loss_type in loss_types:
                model_path = "checkpoints/vae_" + loss_type + "_" + optimizer_type + "_" + scheduler_type
                # Model initialization
                vae = Autoencoder.load_from_checkpoint(model_path,
                                                       encoder=VariationalAutoencoder(1280),
                                                       decoder=EfficientNetDecoder(),
                                                       loss_type=loss_type,
                                                       optimizer_type=optimizer_type,
                                                       scheduler_type=scheduler_type,
                                                       step_size=0,
                                                       lr=lr,
                                                       wd=wd,
                                                       device="cuda:0",
                                                       batch_size=batch_size)
                srinivasan_model = ClsModel(encoder=copy.deepcopy(vae.encoder), classes=srinivasan_classes, lr=lr,
                                            wd=wd)
                # Set up logging for training progress
                srinivasan_tb_logger = TensorBoardLogger(
                    save_dir=os.path.join(model_path.replace("checkpoints", "checkpoints_srinivasan")))
                # Define early stopping criteria
                monitor = "val_loss"
                mode = "min"
                srinivasan_early_stopping = EarlyStopping(monitor=monitor, patience=patience, verbose=False, mode=mode)
                # Initialize the trainer and start training
                trainer = pl.Trainer(
                    strategy="ddp",
                    default_root_dir=model_path,
                    accelerator="gpu",
                    devices=4,
                    max_epochs=max_epochs,
                    callbacks=[
                        srinivasan_early_stopping,
                        ModelCheckpoint(dirpath=model_path, filename="srinivasan-{epoch}-{val_loss:.2f}", save_top_k=1,
                                        save_weights_only=True, mode=mode, monitor=monitor),
                    ],
                    logger=[srinivasan_tb_logger],
                )
                trainer.fit(srinivasan_model, srinivasan_datamodule)
                ### OCT500
                oct500_model = ClsModel(encoder=copy.deepcopy(vae.encoder), classes=oct500_classes, lr=lr,
                                        wd=wd)
                # Set up logging for training progress
                oct500_tb_logger = TensorBoardLogger(
                    save_dir=os.path.join(model_path.replace("checkpoints", "checkpoints_oct500")))
                # Define early stopping criteria
                oct500_early_stopping = EarlyStopping(monitor=monitor, patience=patience, verbose=False, mode=mode)
                # Initialize the trainer and start training
                trainer = pl.Trainer(
                    strategy="ddp",
                    default_root_dir=model_path,
                    accelerator="gpu",
                    devices=4,
                    max_epochs=max_epochs,
                    callbacks=[
                        oct500_early_stopping,
                        ModelCheckpoint(dirpath=model_path, filename="oct500-{epoch}-{val_loss:.2f}", save_top_k=1,
                                        save_weights_only=True, mode=mode, monitor=monitor),
                    ],
                    logger=[oct500_tb_logger],
                )
                trainer.fit(oct500_model, oct500_datamodule)
