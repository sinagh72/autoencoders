import lightning.pytorch as pl
from torch.utils.data import DataLoader

from oct_dataset import OCTDataset, get_kermany_imgs, get_srinivasan_imgs


class SrinivasanDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, classes: list, split=None, train_transform=None,
                 test_transform=None, num_workers=10):
        super().__init__()
        if split is None:
            split = [1, 0, 0]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.split = split  # split percentage of training, validation, testing, the sum should be 1
        self.classes = classes
        self.workers = num_workers
        self.files = [range(1, 16)]

    def prepare_data(self):
        train_subj = self.files[:int(self.files * self.split[0])]
        val_subj = self.files[int(self.files * self.split[0]):int(self.files * self.split[1])]
        test_subj = self.files[int(self.files * self.split[0])+int(self.files * self.split[1]):]
        # img_paths is a list of lists
        self.train_imgs = get_srinivasan_imgs(data_dir=self.data_dir, ignore_folders=train_subj, classes=self.classes)
        self.val_imgs = get_srinivasan_imgs(data_dir=self.data_dir, ignore_folders=val_subj, classes=self.classes)
        self.test_imgs = get_srinivasan_imgs(data_dir=self.data_dir, ignore_folders=test_subj, classes=self.classes)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, data_dir=self.data_dir,
                                         img_paths=self.train_imgs)
            print("train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.train_transform, data_dir=self.data_dir,
                                       img_paths=self.val_imgs)
            print("val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, data_dir=self.data_dir,
                                        img_paths=self.test_imgs)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=False, drop_last=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.workers)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.workers)


class KermanyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, classes: list, split=None, train_transform=None,
                 test_transform=None, num_workers=10):
        super().__init__()
        if split is None:
            split = [1]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.split = split  # split percentage of training, validation, testing, the sum shoudl be 1
        self.classes = classes
        self.workers = num_workers

    def prepare_data(self, ):
        # img_paths is a list of lists
        self.img_paths = get_kermany_imgs(data_dir=self.data_dir, split=self.split, classes=self.classes)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, data_dir=self.data_dir,
                                         img_paths=self.img_paths[0])
            print("train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.train_transform, data_dir=self.data_dir,
                                       img_paths=self.img_paths[1])
            print("val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, data_dir=self.data_dir,
                                        img_paths=self.img_paths[2])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=False, drop_last=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.workers)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.workers)
