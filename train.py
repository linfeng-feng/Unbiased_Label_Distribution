# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import PhaseModel, Phase2Model, SS2net
import config
from dataset import PhaseDataset, Phase2Dataset, SS2Dataset
import sys
import os
from glob import glob

def file_name(s):
    basename = os.path.basename(s).split(sep='.')[0]
    speaker = basename.split(sep='_')[0]
    return int(speaker)

def train():
    torch.manual_seed(config.seed)
    pl.seed_everything(config.seed)
    # ckpt_path = '/home/lffeng/ssl_label_coding/linear1_ss/checkpoints/best-v3.ckpt'

    dir_tr   = '/home/lffeng/datasets_simu/Libri_Circular_4/train2/feats'
    dir_val  = '/home/lffeng/datasets_simu/Libri_Circular_4/val2/feats'
    # dir_tr   = '/home/lffeng/datasets_simu/Libri_Linear_4/train/feats'
    # dir_val  = '/home/lffeng/datasets_simu/Libri_Linear_4/val/feats''
    list_tr = glob(os.path.join(dir_tr, '*.pt'))
    list_val = glob(os.path.join(dir_val, '*.pt'))
    print(f'The grid resolution is {config.cell_reso}')
    print("number of training samples:", len(list_tr))
    print("number of validation samples:", len(list_val))
    # print(f'sigma = {config.sigma}, theta = {config.theta}')

    train_set = SS2Dataset(list_tr, reso=config.cell_reso, sigma=config.sigma, is_train=True)
    val_set = SS2Dataset(list_val, reso=config.cell_reso, sigma=config.sigma, is_train=False)
    # train_set = SS2Dataset(list_tr, reso=config.cell_reso, sigma=config.sigma, is_train=True)
    # val_set = SS2Dataset(list_val, reso=config.cell_reso, sigma=config.sigma, is_train=False)

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False, num_workers=8)
    
    model = SS2net()
    # checkpoint = torch.load('/home/lffeng/ssl_label_coding/linear1_ss/checkpoints/uld_bce.ckpt')
    # model.load_state_dict(checkpoint['state_dict'])
    
    ckpt_cb = ModelCheckpoint(
        monitor='val_mae', 
        mode='min', 
        dirpath=sys.path[0]+f'/{config.space}/checkpoints/', 
        filename='best',
        save_last=False,
        )
    
    es = EarlyStopping(
        monitor='val_loss', 
        patience=config.patience_stop, 
        mode='min',
        )
    
    Callbacks = [es, ckpt_cb]

    Logger = TensorBoardLogger(
        save_dir=sys.path[0]+f'/{config.space}/logs/', 
        name=config.model_type,
        )
    
    trainer = pl.Trainer(
        max_epochs=config.epochs_max,
        devices=config.gpus, 
        # precision=16,
        callbacks=Callbacks,
        logger=Logger,
        accelerator="gpu",
        strategy=config.strategy,
        num_sanity_val_steps=0,
        )
    
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                # ckpt_path=ckpt_path,
                )


if __name__ == "__main__":
    train()
