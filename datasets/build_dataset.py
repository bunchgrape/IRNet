# Copyright 2022 CircuitNet. All rights reserved.

from torch.utils.data import DataLoader, random_split
import datasets
import time
import torch
from .augmentation import Flip, Rotation
from torchvision.transforms import v2


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            # time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            # data = next(self.iter_loader)
            raise StopIteration()
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self


def build_dataset(opt):
    aug_methods = {"Flip": Flip(), "Rotation": Rotation(**opt)}
    pipeline = [aug_methods[i] for i in opt["aug_pipeline"]] if "aug_pipeline" in opt and not opt["test_mode"] else None
    dataset = datasets.__dict__[opt["dataset_type"]](**opt, pipeline=pipeline)
    if opt["test_mode"]:
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        return IterLoader(DataLoader(dataset=dataset, num_workers=6, batch_size=opt["batch_size"], shuffle=True, drop_last=True, pin_memory=True))
