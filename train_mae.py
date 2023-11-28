import os
import lightning.pytorch as pl
from dotenv import load_dotenv
from DataModule import KermanyDataModule
from models.AE import Autoencoder
from models.mae_draft import MaskedAutoencoder, EfficientNetDecoder
from utils.transformations import get_train_transformation
from utils.util import set_seed
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers

if __name__ == "__main__":
    max_epochs = 100
    set_seed(10)
    model_path = "checkpoints/mae"

    kermany_classes = [("NORMAL", 0),
                       ("DRUSEN", 1),
                       ("DME", 2),
                       ("CNV", 3),
                       ]
    img_size = 256
    batch_size = 24
    load_dotenv(dotenv_path="./config/.env")
    dataset_path = os.getenv('KERMANY')
    kermany_datamodule = KermanyDataModule(data_dir=dataset_path,
                                           batch_size=batch_size,
                                           classes=kermany_classes,
                                           split=[0.95, 0.05],
                                           train_transform=get_train_transformation(size=img_size)
                                           )
    # preparing config
    kermany_datamodule.prepare_data()
    kermany_datamodule.setup("train")
    kermany_datamodule.setup("val")
    train_loader = kermany_datamodule.train_dataloader()
    val_loader = kermany_datamodule.val_dataloader()

    loss_type = "mse"
    optimizer_type = "SGD"
    scheduler_type = "StepLR"
    patience = 10
    step_size = len(train_loader)//batch_size * patience
    model_path += "_" + loss_type + "_" + optimizer_type
    lr = 1e-4
    wd = 1e-6
    # Model initialization
    model = Autoencoder(encoder=MaskedAutoencoder(),
                        decoder=EfficientNetDecoder(),
                        loss_type=loss_type,
                        optimizer_type=optimizer_type,
                        scheduler_type=scheduler_type,
                        step_size=step_size,
                        lr=lr,
                        wd=wd,
                        device="cuda:0")
    # Set up logging for training progress
    csv_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "csv_log/"))
    tb_logger = TensorBoardLogger(save_dir=os.path.join(model_path, "tb_log/"), name="mce")
    # Define early stopping criteria
    monitor = "val_loss"
    mode = "min"
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, verbose=False, mode=mode)
    # Initialize the trainer and start training
    trainer = pl.Trainer(
        default_root_dir=model_path,
        accelerator="gpu",
        devices=[0],
        max_epochs=max_epochs,
        callbacks=[
            early_stopping,
            ModelCheckpoint(dirpath=model_path, filename="mce-{epoch}-{val_loss:.2f}", save_top_k=1,
                            save_weights_only=True, mode=mode, monitor=monitor),
        ],
        logger=[tb_logger, csv_logger],
    )
    trainer.fit(model, datamodule=kermany_datamodule)
