import copy
import glob
import json
import os
import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from DataModule import SrinivasanDataModule, OCT500DataModule
from models.AE import Autoencoder
from models.base import ClsModel
from models.cae import get_EfficientNetEncoder, EfficientNetDecoder
from utils.transformations import get_train_transformation, get_test_transformation
from utils.util import set_seed
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers

if __name__ == "__main__":
    max_epochs = 5
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
                                                 split=[0.7, 0, 0.3],
                                                 train_transform=get_train_transformation(size=img_size),
                                                 test_transform=get_test_transformation(size=img_size),
                                                 num_workers=torch.cuda.device_count() * 2
                                                 )
    # preparing config
    srinivasan_datamodule.prepare_data()
    srinivasan_datamodule.setup("train")
    srinivasan_datamodule.setup("test")
    srinivasan_datamodule.setup("val")
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
    oct500_datamodule.setup("val")
    oct500_train_loader = oct500_datamodule.train_dataloader()

    loss_types = "mse", "cosine"
    optimizer_types = ["SGD", "SGD_nesterov", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Nadam"]
    scheduler_types = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    lr = 1e-4
    wd = 1e-6
    save_path = "cae_checkpoints/"
    for scheduler_type in scheduler_types:
        for optimizer_type in optimizer_types:
            for loss_type in loss_types:
                model_type = "cae_" + loss_type + "_" + optimizer_type + "_" + scheduler_type
                # Model initialization
                model_path = glob.glob(os.path.join("checkpoints/"+model_type, "mce*.ckpt"))[0]
                cae = Autoencoder.load_from_checkpoint(model_path,
                                                       encoder=get_EfficientNetEncoder(),
                                                       decoder=EfficientNetDecoder(),
                                                       loss_type=loss_type,
                                                       optimizer_type=optimizer_type,
                                                       scheduler_type=scheduler_type,
                                                       step_size=0,
                                                       lr=lr,
                                                       wd=wd,
                                                       device="cuda:0",
                                                       batch_size=batch_size)
                srinivasan_model = ClsModel(encoder=copy.deepcopy(cae.encoder), classes=srinivasan_classes, lr=lr,
                                            wd=wd)
                # Set up logging for training progress
                srinivasan_tb_logger = TensorBoardLogger(
                    save_dir=os.path.join(save_path, "srinivasan", model_type))
                # Define early stopping criteria
                monitor = "val_loss"
                mode = "min"
                # Initialize the trainer and start training
                trainer = pl.Trainer(
                    strategy="ddp",
                    default_root_dir=os.path.join(save_path, "srinivasan", model_type),
                    accelerator="gpu",
                    devices=4,
                    max_epochs=max_epochs,
                    logger=[srinivasan_tb_logger],
                    log_every_n_steps=1
                )
                trainer.fit(srinivasan_model, srinivasan_datamodule)
                res = json.dumps(trainer.test(srinivasan_model, srinivasan_datamodule.test_dataloader()))
                f = open(os.path.join(save_path, "srinivasan", model_type+"_results.json"), "w")
                f.write(res)
                f.close()

                """OCT-500"""
                # oct500_model = ClsModel(encoder=copy.deepcopy(cae.encoder), classes=oct500_classes, lr=lr,
                #                         wd=wd)
                # # Set up logging for training progress
                # oct500_tb_logger = TensorBoardLogger(
                #     save_dir=os.path.join(save_path, "oct500", model_type))
                # # Initialize the trainer and start training
                # trainer = pl.Trainer(
                #     strategy="ddp",
                #     default_root_dir=os.path.join(save_path, "oct500", model_type),
                #     accelerator="gpu",
                #     devices=4,
                #     max_epochs=max_epochs,
                #     logger=[oct500_tb_logger],
                #     log_every_n_steps=1
                # )
                # trainer.fit(oct500_model, oct500_datamodule)
                # res = json.dumps(trainer.test(oct500_model, oct500_datamodule.test_dataloader()))
                # f = open(os.path.join(save_path, "oct500", model_type + "_results.json"), "w")
                # f.write(res)
                # f.close()

