import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from Data.test_dataset import TestDataset
from Data.train_dataset import TrainDataset
from Data.val_dataset import ValDataset


class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, seg_length: float, train_n_workers: int, n_workers: int = 1,
                 sample_rate: int = 16000):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seg_length = seg_length
        self.train_n_workers = train_n_workers
        self.n_workers = n_workers
        self.sample_rate = sample_rate
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.save_hyperparameters()
        return

    def setup(self, stage: str) -> None:
        self.train_dataset = TrainDataset(self.data_dir, self.seg_length, self.sample_rate)
        self.val_dataset = ValDataset(self.data_dir, self.seg_length, self.sample_rate)
        self.test_dataset = TestDataset(self.data_dir, self.sample_rate)
        return

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.train_n_workers,
                          shuffle=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.n_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.n_workers)
