# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from model import PhaseModel, SS2net
import config
from dataset import PhaseDataset, SS2Dataset
import os
from glob import glob


def file_name(s):
    basename = os.path.basename(s)[:-3]
    return int(basename)

def test():
    # dir_test = '/home/lffeng/datasets/Libri_adhoc_nodes10_splited/room2/feats'
    dir_test = '/home/lffeng/datasets/Libri_adhoc_nodes10_splited/room2/test2/feats'
    
    list_test = glob(os.path.join(dir_test, '*.pt'))
    # list_test.sort(key=file_name)
    print("number of testing samples:", len(list_test))

    test_set = PhaseDataset(list_test, config.cell_reso, config.sigma, is_train=False)

    test_loader = DataLoader(dataset=test_set, batch_size=50, shuffle=False, num_workers=8)
    
    # model_path = f'/home/lffeng/ssl_label_coding/linear2/checkpoints/uld_msewo_{config.cell_reso}.ckpt'
    model_path = '/home/lffeng/ssl_label_coding/circular1/checkpoints/1hot_72.ckpt'

    print(model_path)
    model = PhaseModel.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        devices=1,
        accelerator="cpu",
        # strategy=config.strategy,
        num_sanity_val_steps=0,
        # fast_dev_run=True
        )
    
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    test()
