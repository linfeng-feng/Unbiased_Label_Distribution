import torch
from torch.utils.data import Dataset
import config
from encoding_decoding import onehot_encoding, unbiased_encoding, gaussian_encoding, soft_encoding



class PhaseDataset(Dataset):
    def __init__(self, total_list, reso, sigma, is_train):
        self.total_list = total_list
        self.reso = reso
        self.sigma = sigma
        self.is_train = is_train
        # if is_train:
        #     self.total_list = self.total_list[:360]

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, item):
        filepath = self.total_list[item]
        data = torch.load(filepath)

        doas = data['doas'] # shape=(num_srcs)
        num_srcs = doas.shape[0]
        doas, _ = torch.sort(doas, dim=0, descending=True)

        stft = data['stft'][:, 1:, :]
        x = torch.angle(stft)
        x = x.permute((2, 0, 1)) # shape=(frame, mic, freq)
        frames_num = x.shape[0]

        y = torch.zeros(num_srcs, self.reso+1)
        for src in range(num_srcs):
            # y[src] = gaussian_encoding(sigma=self.sigma, loc=doas[src], reso=self.reso)
            # y[src] = onehot_encoding(loc=doas[src], reso=self.reso)
            # y[src] = soft_encoding(loc=doas[src], reso=self.reso)
            y[src] = unbiased_encoding(loc=doas[src], reso=self.reso)
        y = y.unsqueeze(1).expand(num_srcs, frames_num, self.reso+1)

        return x, y, doas


class Phase2Dataset(Dataset):
    def __init__(self, total_list, reso, sigma, is_train):
        self.total_list = total_list
        self.reso = reso
        self.sigma = sigma
        self.is_train = is_train
        # if is_train:
        #     self.total_list = self.total_list[:360]

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, item):
        filepath = self.total_list[item]
        data = torch.load(filepath)

        doas = data['doas'] # shape=(num_srcs)
        num_srcs = doas.shape[0]
        doas, _ = torch.sort(doas, dim=0, descending=True)

        stft = data['stft'][:, 1:, :]
        x = torch.angle(stft)
        x = x.permute((2, 0, 1)) # shape=(frame, mic, freq)

        y = torch.zeros(num_srcs, self.reso+1)
        y2 = torch.zeros(num_srcs, self.reso+1)
        for src in range(num_srcs):
            # y[src] = gaussian_encoding(sigma=self.sigma, loc=doas[src], reso=self.reso)
            # y[src] = onehot_encoding(loc=doas[src], reso=self.reso)
            # y[src] = soft_encoding(loc=doas[src], reso=self.reso)
            y2[src] = unbiased_encoding(loc=doas[src], reso=self.reso)

        # return x, y, doas
        return x, y, y2, doas

class SSDataset(Dataset):
    def __init__(self, total_list, reso, sigma, is_train):
        self.total_list = total_list
        self.reso = reso
        self.sigma = sigma
        self.is_train = is_train
        # if is_train:
        #     self.total_list = self.total_list[:360]

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, item):
        filepath = self.total_list[item]
        data = torch.load(filepath)

        doas = data['doas'] # shape=(num_srcs)
        num_srcs = doas.shape[0]
        doas, _ = torch.sort(doas, dim=0, descending=True)

        stft = data['stft'][:, 1:, :]
        x = torch.cat((stft.real, stft.imag), dim=0)
        x = x.view(x.shape[0], x.shape[1], -1, 7)   # (Channel=8, Freq=256, -1, 7)
        x = x.permute((2, 0, 3, 1)) # (frames_num, channel, frame, freq)
        frames_num = x.shape[0]

        y = torch.zeros(num_srcs, self.reso+1)
        for src in range(num_srcs):
            # y[src] = gaussian_encoding(sigma=self.sigma, loc=doas[src], reso=self.reso)
            # y[src] = onehot_encoding(loc=doas[src], reso=self.reso)
            # y[src] = soft_encoding(loc=doas[src], reso=self.reso)
            y[src] = unbiased_encoding(loc=doas[src], reso=self.reso)
        y = y.unsqueeze(1).expand(num_srcs, frames_num, self.reso+1)

        return x, y, doas
    

class SS2Dataset(Dataset):
    def __init__(self, total_list, reso, sigma, is_train):
        self.total_list = total_list
        self.reso = reso
        self.sigma = sigma
        self.is_train = is_train
        # if is_train:
        #     self.total_list = self.total_list[:360]

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, item):
        filepath = self.total_list[item]
        data = torch.load(filepath)

        doas = data['doas'] # shape=(num_srcs)
        num_srcs = doas.shape[0]
        doas, _ = torch.sort(doas, dim=0, descending=True)

        stft = data['stft'][:, 1:, :]
        x = torch.cat((stft.real, stft.imag), dim=0)
        x = x.view(x.shape[0], x.shape[1], -1, 7)   # (Channel=8, Freq=256, -1, 7)
        x = x.permute((2, 0, 3, 1)) # (frames_num, channel, frame, freq)
        frames_num = x.shape[0]

        y = torch.zeros(num_srcs, self.reso+1)
        for src in range(num_srcs):
            # y[src] = gaussian_encoding(sigma=self.sigma, loc=doas[src], reso=self.reso)
            y[src] = onehot_encoding(loc=doas[src], reso=self.reso)
            # y[src] = soft_encoding(loc=doas[src], reso=self.reso)
            # y[src] = unbiased_encoding(loc=doas[src], reso=self.reso)

        return x, y, doas



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    doas = torch.tensor([92.4])
    num_srcs = len(doas)
    reso = 180
    y = torch.zeros(num_srcs, reso+1)
    for src in range(num_srcs):
        y[src] = gaussian_encoding(sigma=8, loc=doas[src], reso=180)
        # y[src] = soft_encoding(loc=doas[src], reso=reso)
        # y[src] = onehot_encoding(loc=doas[src], reso=reso)
        # y[src] = unbiased_encoding(loc=doas[src], reso=reso)
        # if src==0:
        #     y[src] = y[src] * 0.9
        # else:
        #     y[src] = y[src] * 0.7
    y, _ = torch.max(y, dim=0)
    # y = y.numpy() * 0.7

    x = np.arange(reso+1)
    plt.plot(x, y, linewidth = 5)
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.xlim(0, 180)
    plt.ylim(0, 1)
    plt.savefig('1', dpi=128)
