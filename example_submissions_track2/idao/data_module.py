import pathlib as path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataloader import IDAODataset, img_loader, InferenceDataset


class IDAODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: path.Path, batch_size: int, cfg):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cfg = cfg


        self.public_dataset = InferenceDataset(
                    main_dir=self.data_dir.joinpath("public_test"),
                    loader=img_loader,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.CenterCrop(120)]
                    ),
                )
        self.private_dataset = InferenceDataset(
                    main_dir=self.data_dir.joinpath("private_test"),
                    loader=img_loader,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.CenterCrop(120)]
                    ),
                )


    def setup(self, stage=None):
        # called on every GPU
        self.train, self.val = random_split(
            self.dataset, [10000, 3404], generator=torch.Generator().manual_seed(666)
        )


    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, 1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            torch.utils.data.ConcatDataset([self.private_dataset, self.public_dataset]),
            1,
            num_workers=0,
            shuffle=False
            )
