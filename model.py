# -*- coding: utf-8 -*-
import torch
torch.set_float32_matmul_precision('high')
from torch import nn, optim, sigmoid, Tensor, abs, floor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment

import config
from loss_func import NLAELoss, WassersteinLoss
from encoding_decoding import weighted_adjacent_decoding as wad


def log(x):
    return torch.clamp(torch.log(x), min=-100)


def hungarian_dist(doa_hs, doas, num_srcs):
    distance_matrix = torch.zeros((num_srcs, num_srcs), device='cpu')
    for i in range(num_srcs):
        for j in range(num_srcs):
            diff = abs(doas[i]-doa_hs[j])
            distance_matrix[i, j] = min(diff, 360-diff)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matched_DOA_pred = doa_hs[row_ind]
    matched_DOA_true = doas[col_ind]
    diff = abs(matched_DOA_true - matched_DOA_pred)
    return torch.where(diff<360-diff, diff, 360-diff).mean()



class PhaseModel(pl.LightningModule):
    def __init__(self):
        super(PhaseModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(2, 1),
                      stride=(1, 1),
                      padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(2, 1),
                      stride=(1, 1),
                      padding=0),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(2, 1),
                      stride=(1, 1),
                      padding=0),
            nn.ReLU(),
        )
        self.fl1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 512),   # 256*64=16384, 256=Nf/2, 64 is out_channels
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fl2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fl3 = nn.Sequential(
            nn.Linear(512, config.cell_reso+1)
        )
        # self.criterion = GibbsWeightedLoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        # self.criterion = CELoss()
        # self.criterion = NLAELoss()
        # self.criterion = WassersteinLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        batchsize, frame_num, mic_num, freq = x.shape
        x = x.view(batchsize*frame_num, 1, mic_num, freq)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fl1(x)
        x = self.fl2(x)
        x = self.fl3(x)
        # x = sigmoid(x)
        x = F.softmax(x, dim=-1)
        return x
    

    def training_step(self, batch, batch_idx):
        x, y, doa = batch
        batchsize, num_srcs, frames_num, num_classes = y.shape
        logits = self.forward(x)
        loss = self.criterion(logits, y.view(-1, num_classes))*num_classes
        # x, doa = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, doa.view(-1, 1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, doa = batch
        batchsize, num_srcs, frames_num, num_classes = y.shape
        logits = self.forward(x)
        loss = self.criterion(logits, y.view(-1, num_classes))*num_classes
        
        # logits = F.softmax(logits, dim=-1)
        # logits = y.view(-1, num_classes)  # 量化误差
        logits = logits.clamp(min=0, max=1)
        logits = logits.view(batchsize*num_srcs, frames_num, num_classes).mean(dim=1)
        doa_hs = torch.zeros((3, batchsize*num_srcs), device=logits.device) # shape[0] is top-1, wad-2 and wad-3

        doa_hs[0, :] = wad(logits, selected_classes=1) * config.cell_len
        doa_hs[1, :] = wad(logits, selected_classes=2) * config.cell_len
        doa_hs[2, :] = wad(logits, selected_classes=3) * config.cell_len

        doa = doa.view(-1)
        mae_sum = []
        for i in range(3):
            doa_h = doa_hs[i]
            diff = torch.abs(doa - doa_h)
            mae_sum.append(torch.min(diff, 360-diff).sum()) # The sum of mae in a batch

        y_h = logits.argmax(dim=-1)  # y_h.shape = (num_srcs*batchsize)
        y = y.view(batchsize*num_srcs, frames_num, num_classes).mean(dim=1)
        y = y.argmax(dim=-1)
        mask = torch.eq(y_h, y)
        n_correct = torch.sum(mask) # The number of samples correctly classified in a batch
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append([len(y), mae_sum, n_correct])
        # x, doa = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, doa.view(-1, 1))
        
        # doa_h = logits.view_as(doa).mean(dim=1)
        # doa = doa.mean(dim=1)
        # diff = torch.abs(doa - doa_h)
        # mae_sum = torch.min(diff, 360-diff).sum() # The sum of mae in a batch
        # self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.validation_step_outputs.append([doa.shape[0], mae_sum])
        
        
    
    def on_validation_epoch_end(self):
        n_total, n_correct, mae_sum_1, mae_sum_2, mae_sum_3,  = 0, 0, 0, 0, 0
        for outputs in self.validation_step_outputs:
            n_total += outputs[0]
            mae_sum_1 += outputs[1][0]
            mae_sum_2 += outputs[1][1]
            mae_sum_3 += outputs[1][2]
            n_correct += outputs[2]
        mae_1 = mae_sum_1 / n_total
        mae_2 = mae_sum_2 / n_total
        mae_3 = mae_sum_3 / n_total
        acc = n_correct / n_total * 100
        self.log('mae_1', mae_1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_2', mae_2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_3', mae_3, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', min(mae_1, mae_2, mae_3), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        # n_total, mae_sum = 0, 0
        # for outputs in self.validation_step_outputs:
        #     n_total += outputs[0]
        #     mae_sum += outputs[1]
        # mae = mae_sum / n_total
        # self.log('val_mae', mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.validation_step_outputs.clear()  # free memory    
    
    test_step = validation_step
    on_test_epoch_end = on_validation_epoch_end
    
    
    def configure_optimizers(self):
            
        optimizer = optim.AdamW(params=self.parameters(), lr=config.learning_rate)
        return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(
                    optimizer, mode='min', verbose=True, factor=config.scheduler_factor, patience=config.patience, min_lr=config.min_lr),
                # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=config.epochs_max),
                'monitor': "val_loss"
                }
    

class Phase2Model(pl.LightningModule):
    def __init__(self):
        super(Phase2Model, self).__init__()
        num_classes = config.cell_reso+1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=4,
                      kernel_size=(2, 1),
                      stride=(1, 1),
                      padding=(0, 0)
                      ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=(2, 3),
                      stride=(1, 1),
                      padding=(0, 1)
                      ),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 3),
                      stride=(1, 1),
                      padding=(0, 1)
                      ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fl1=nn.Sequential(
            nn.Linear(8192, num_classes*2),   # 256*64=16384, 256=Nf/2, 64 is out_channels
            nn.BatchNorm1d(num_classes*2),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.mask1 = nn.LSTM(
            input_size = num_classes*2,
            hidden_size = num_classes,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.fl2 = nn.Sequential(
            nn.Linear(num_classes*2, num_classes)
        )
        
        
        # self.criterion = GibbsWeightedLoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()
        # self.criterion = NLAELoss()
        # self.criterion = WassersteinLoss()

        self.validation_step_outputs = []

    def forward(self, x):
        batchsize, frame_num, mic_num, freq = x.shape
        x = x.view(batchsize*frame_num, 1, mic_num, freq)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fl1(x)

        w1, _ = self.mask1(x)
        w1 = sigmoid(w1)
        w2 = 1 - w1
        w1 = w1.view(batchsize, frame_num, -1)
        w2 = w2.view(batchsize, frame_num, -1)
        x = x.view(batchsize, frame_num, -1)
        x1 = (w1*x).sum(dim=1, keepdim=True) / w1.sum(dim=1, keepdim=True)
        x2 = (w2*x).sum(dim=1, keepdim=True) / w2.sum(dim=1, keepdim=True)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(-1, x.shape[-1])

        x = self.fl2(x)
        x = F.softmax(x, dim=-1)
        # x = sigmoid(x)
        return x
    

    def training_step(self, batch, batch_idx):
        x, y, y2, doa = batch
        logits = self.forward(x)
        batchsize, num_srcs, num_classes = y.shape
        y = y.view(-1, num_classes)
        # loss = self.criterion(logits, y)
        y2 = y2.view(-1, num_classes)
        loss = (self.criterion(logits, y)*(1-config.alpha)+self.criterion(logits, y2)*config.alpha)*(config.cell_reso+1)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y, y2, doa = batch
        logits = self.forward(x)
        batchsize, num_srcs, num_classes = y.shape
        y = y.view(-1, num_classes)
        # loss = self.criterion(logits, y)
        y2 = y2.view(-1, num_classes)
        loss = (self.criterion(logits, y)*(1-config.alpha)+self.criterion(logits, y2)*config.alpha)*(config.cell_reso+1)
        
        # logits = F.softmax(logits, dim=-1)
        logits = logits.clamp(min=0, max=1) 
        doa_hs = torch.zeros((3, batchsize*num_srcs), device=logits.device) # shape[0] is top-1, wad-2 and wad-3

        doa_hs[0, :] = wad(logits, selected_classes=1) * config.cell_len
        doa_hs[1, :] = wad(logits, selected_classes=2) * config.cell_len
        doa_hs[2, :] = wad(logits, selected_classes=3) * config.cell_len

        doa = doa.view(-1)
        mae_sum = []
        for i in range(3):
            doa_h = doa_hs[i]
            diff = torch.abs(doa - doa_h)
            mae_sum.append(torch.min(diff, 360-diff).sum()) # The sum of mae in a batch

        y_h = logits.argmax(dim=-1)  # y_h.shape = (num_srcs*batchsize)
        y = y.argmax(dim=-1)
        mask = torch.eq(y_h, y)
        n_correct = torch.sum(mask) # The number of samples correctly classified in a batch
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append([len(y), mae_sum, n_correct])


    def on_validation_epoch_end(self):
        n_total, n_correct, mae_sum_1, mae_sum_2, mae_sum_3,  = 0, 0, 0, 0, 0
        for outputs in self.validation_step_outputs:
            n_total += outputs[0]
            mae_sum_1 += outputs[1][0]
            mae_sum_2 += outputs[1][1]
            mae_sum_3 += outputs[1][2]
            n_correct += outputs[2]
        mae_1 = mae_sum_1 / n_total
        mae_2 = mae_sum_2 / n_total
        mae_3 = mae_sum_3 / n_total
        acc = n_correct / n_total * 100
        self.log('mae_1', mae_1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_2', mae_2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_3', mae_3, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', min(mae_1, mae_2, mae_3), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    test_step = validation_step
    on_test_epoch_end = on_validation_epoch_end
    
    
    def configure_optimizers(self):
            
        optimizer = optim.AdamW(params=self.parameters(), lr=config.learning_rate)
        return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(
                    optimizer, mode='min', verbose=True, factor=config.scheduler_factor, patience=config.patience, min_lr=config.min_lr),
                # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=config.epochs_max),
                'monitor': "val_loss"
                } 


class SSnet(pl.LightningModule):
    def __init__(self):
        super(SSnet, self).__init__()

        # the input shape should be: (Channel=8, Time=7, Freq=337)
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, (1, 7), (1, 3), (0, 0)), # (32, 7, 110)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, (1, 5), (1, 2), (0, 0)), # (32, 7, 52)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128, affine=False), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, config.cell_reso+1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(config.cell_reso+1), nn.ReLU(inplace=True)
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(40, 500, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(500), nn.ReLU(inplace=True)
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(500, 1, kernel_size=(7, 5), stride=(1, 1), padding=(0, 2)),
            # nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

        # self.criterion = GibbsWeightedLoss()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()
        # self.criterion = NLAELoss()
        # self.criterion = WassersteinLoss()
        self.validation_step_outputs = []
        

    def forward(self, x):
        batchsize, frame_num, mic_num, frame, freq = x.shape
        x = x.view(batchsize*frame_num, mic_num, frame, freq)
        # a.shape: [B, 8, 7, 337]
        x = self.conv1(x) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_1(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_2(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_3(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_4(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_5(x)) # [B, 128, 7, 54]
        x = self.conv_6(x) # [B, 360, 7, 54]
        x = x.permute(0, 3, 2, 1) # [B, 54, 7, 360]
        x = self.conv_7(x) # [B, 500, 7, 360]
        x = self.conv_8(x) # [B, 1, 1, 360]
        x = x.view(x.size(0), -1) # [B, 360]
        # x = sigmoid(x)
        x = F.softmax(x, dim=-1)

        return x
    

    def training_step(self, batch, batch_idx):
        x, y, doa = batch
        batchsize, num_srcs, frames_num, num_classes = y.shape
        logits = self.forward(x)
        loss = self.criterion(logits, y.view(-1, num_classes))
        # x, doa = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, doa.view(-1, 1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, doa = batch
        batchsize, num_srcs, frames_num, num_classes = y.shape
        logits = self.forward(x)
        loss = self.criterion(logits, y.view(-1, num_classes))
        
        # logits = F.softmax(logits, dim=-1)
        # logits = y.view(-1, num_classes)  # 量化误差
        logits = logits.clamp(min=0, max=1)
        logits = logits.view(batchsize*num_srcs, frames_num, num_classes).mean(dim=1)
        doa_hs = torch.zeros((3, batchsize*num_srcs), device=logits.device) # shape[0] is top-1, wad-2 and wad-3

        doa_hs[0, :] = wad(logits, selected_classes=1) * config.cell_len
        doa_hs[1, :] = wad(logits, selected_classes=2) * config.cell_len
        doa_hs[2, :] = wad(logits, selected_classes=3) * config.cell_len

        doa = doa.view(-1)
        mae_sum = []
        for i in range(3):
            doa_h = doa_hs[i]
            diff = torch.abs(doa - doa_h)
            mae_sum.append(torch.min(diff, 360-diff).sum()) # The sum of mae in a batch

        y_h = logits.argmax(dim=-1)  # y_h.shape = (num_srcs*batchsize)
        y = y.view(batchsize*num_srcs, frames_num, num_classes).mean(dim=1)
        y = y.argmax(dim=-1)
        mask = torch.eq(y_h, y)
        n_correct = torch.sum(mask) # The number of samples correctly classified in a batch
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append([len(y), mae_sum, n_correct])
        # x, doa = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, doa.view(-1, 1))
        
        # doa_h = logits.view_as(doa).mean(dim=1)
        # doa = doa.mean(dim=1)
        # diff = torch.abs(doa - doa_h)
        # mae_sum = torch.min(diff, 360-diff).sum() # The sum of mae in a batch
        # self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.validation_step_outputs.append([doa.shape[0], mae_sum])
        
        
    
    def on_validation_epoch_end(self):
        n_total, n_correct, mae_sum_1, mae_sum_2, mae_sum_3,  = 0, 0, 0, 0, 0
        for outputs in self.validation_step_outputs:
            n_total += outputs[0]
            mae_sum_1 += outputs[1][0]
            mae_sum_2 += outputs[1][1]
            mae_sum_3 += outputs[1][2]
            n_correct += outputs[2]
        mae_1 = mae_sum_1 / n_total
        mae_2 = mae_sum_2 / n_total
        mae_3 = mae_sum_3 / n_total
        acc = n_correct / n_total * 100
        self.log('mae_1', mae_1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_2', mae_2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_3', mae_3, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', min(mae_1, mae_2, mae_3), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        # n_total, mae_sum = 0, 0
        # for outputs in self.validation_step_outputs:
        #     n_total += outputs[0]
        #     mae_sum += outputs[1]
        # mae = mae_sum / n_total
        # self.log('val_mae', mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.validation_step_outputs.clear()  # free memory    
    
    test_step = validation_step
    on_test_epoch_end = on_validation_epoch_end
    
    
    def configure_optimizers(self):
            
        optimizer = optim.AdamW(params=self.parameters(), lr=config.learning_rate)
        return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(
                    optimizer, mode='min', verbose=True, factor=config.scheduler_factor, patience=config.patience, min_lr=config.min_lr),
                # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=config.epochs_max),
                'monitor': "val_loss"
                }
    


class SS2net(pl.LightningModule):
    def __init__(self):
        super(SS2net, self).__init__()
        num_classes = config.cell_reso+1
        # the input shape should be: (Channel=8, Time=7, Freq=256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, (1, 7), (1, 3), (0, 0)), # (32, 7, 84)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, (1, 5), (1, 2), (0, 0)), # (32, 7, 40)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128, affine=False), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )

        # self.conv_6 = nn.Sequential(
        #     nn.Conv2d(128, config.cell_reso+1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(config.cell_reso+1), nn.ReLU(inplace=True)
        # )

        # self.conv_7 = nn.Sequential(
        #     nn.Conv2d(40, 500, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(500), nn.ReLU(inplace=True)
        # )

        # self.conv_8 = nn.Sequential(
        #     nn.Conv2d(500, 1, kernel_size=(7, 5), stride=(1, 1), padding=(0, 2)),
        #     # nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        # )

        self.fl1=nn.Sequential(
            nn.Linear(35840, num_classes*2),   # 256*64=16384, 256=Nf/2, 64 is out_channels
            nn.BatchNorm1d(num_classes*2),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fl2 = nn.Sequential(
            nn.Linear(num_classes*2, num_classes)
        )
        self.mask1 = nn.LSTM(
            input_size = num_classes*2,
            hidden_size = num_classes,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        self.relu = nn.ReLU(inplace=True)

        # self.criterion = GibbsWeightedLoss()
        # self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()
        self.criterion = NLAELoss()
        # self.criterion = WassersteinLoss()
        self.validation_step_outputs = []
        

    def forward(self, x):
        batchsize, frame_num, mic_num, frame, freq = x.shape
        x = x.view(batchsize*frame_num, mic_num, frame, freq)
        # a.shape: [B, 8, 7, 337]
        x = self.conv1(x) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_1(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_2(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_3(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_4(x)) # [B, 128, 7, 54]
        x = self.relu(x+self.conv_5(x)) # [B, 128, 7, 54]
        x = x.view(batchsize*frame_num, -1)
        x = self.fl1(x)

        w1, _ = self.mask1(x)
        w1 = sigmoid(w1)
        w2 = 1 - w1
        w1 = w1.view(batchsize, frame_num, -1)
        w2 = w2.view(batchsize, frame_num, -1)
        x = x.view(batchsize, frame_num, -1)
        x1 = (w1*x).sum(dim=1, keepdim=True) / w1.sum(dim=1, keepdim=True)
        x2 = (w2*x).sum(dim=1, keepdim=True) / w2.sum(dim=1, keepdim=True)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(-1, x.shape[-1])

        x = self.fl2(x)
        x = F.softmax(x, dim=-1)
        # x = sigmoid(x)

        return x
    

    def training_step(self, batch, batch_idx):
        x, y, doa = batch
        logits = self.forward(x)
        batchsize, num_srcs, num_classes = y.shape
        y = y.view(-1, num_classes)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y, doa = batch
        logits = self.forward(x)
        batchsize, num_srcs, num_classes = y.shape
        y = y.view(-1, num_classes)
        loss = self.criterion(logits, y)
        
        # logits = F.softmax(logits, dim=-1)
        logits = logits.clamp(min=0, max=1) 
        doa_hs = torch.zeros((3, batchsize*num_srcs), device=logits.device) # shape[0] is top-1, wad-2 and wad-3

        doa_hs[0, :] = wad(logits, selected_classes=1) * config.cell_len
        doa_hs[1, :] = wad(logits, selected_classes=2) * config.cell_len
        doa_hs[2, :] = wad(logits, selected_classes=3) * config.cell_len

        doa = doa.view(-1)
        mae_sum = []
        for i in range(3):
            doa_h = doa_hs[i]
            diff = torch.abs(doa - doa_h)
            mae_sum.append(torch.min(diff, 360-diff).sum()) # The sum of mae in a batch

        y_h = logits.argmax(dim=-1)  # y_h.shape = (num_srcs*batchsize)
        y = y.argmax(dim=-1)
        mask = torch.eq(y_h, y)
        n_correct = torch.sum(mask) # The number of samples correctly classified in a batch
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append([len(y), mae_sum, n_correct])

    def on_validation_epoch_end(self):
        n_total, n_correct, mae_sum_1, mae_sum_2, mae_sum_3,  = 0, 0, 0, 0, 0
        for outputs in self.validation_step_outputs:
            n_total += outputs[0]
            mae_sum_1 += outputs[1][0]
            mae_sum_2 += outputs[1][1]
            mae_sum_3 += outputs[1][2]
            n_correct += outputs[2]
        mae_1 = mae_sum_1 / n_total
        mae_2 = mae_sum_2 / n_total
        mae_3 = mae_sum_3 / n_total
        acc = n_correct / n_total * 100
        self.log('mae_1', mae_1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_2', mae_2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mae_3', mae_3, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', min(mae_1, mae_2, mae_3), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    test_step = validation_step
    on_test_epoch_end = on_validation_epoch_end
    
    
    def configure_optimizers(self):
            
        optimizer = optim.AdamW(params=self.parameters(), lr=config.learning_rate)
        return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(
                    optimizer, mode='min', verbose=True, factor=config.scheduler_factor, patience=config.patience, min_lr=config.min_lr),
                # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=config.epochs_max),
                'monitor': "val_loss"
                } 



if __name__ == "__main__":
    x = torch.randn(1, 1, 4, 256)
    model = PhaseModel()
    y = model(x)
    print(y.shape)