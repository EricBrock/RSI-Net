import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class Color_BGR_Data(Dataset):
    def __init__(self, img_path, normalzero2one=True, has_name=False):
        self.img_path = img_path
        self.img_list = os.listdir(self.img_path)
        self.normalzero2one = normalzero2one
        self.has_name = has_name

    def __getitem__(self, index):
        img_name = self.img_list[index]

        img_bgr = cv2.imread(os.path.join(self.img_path, img_name), 1)
        img_gray = cv2.imread(os.path.join(self.img_path, img_name), 0)
        np.expand_dims(img_gray, 0).repeat(3, axis=0)

        img_bgr= img_bgr.transpose((2, 0, 1))

        if self.normalzero2one:
            img_bgr, img_bgr = img_bgr / 255., img_bgr / 255.

        if self.has_name:
            return img_bgr, img_bgr, img_name
        else:
            return img_bgr, img_bgr

    def __len__(self):
        return len(self.img_list)


class Color_BGR_Data_Loader():
    def __init__(self, img_path, batch_size, normalzero2one=False, has_name=False, shuf=True):
        self.dataset = Color_BGR_Data(img_path, normalzero2one=normalzero2one, has_name=has_name)
        self.batch = batch_size
        self.shuf = shuf

    def loader(self):
        loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch, shuffle=self.shuf,num_workers=0, drop_last=True)
        return loader


class SR_BGR_Data(Dataset):
    def __init__(self, img_path, normalzero2one=True, has_name=False, sr_factor=1):
        self.img_path = img_path
        self.img_list = os.listdir(self.img_path)
        self.normalzero2one = normalzero2one
        self.has_name = has_name
        self.sr_factor = sr_factor

    def __getitem__(self, index):
        img_name = self.img_list[index]

        img_bgr_hr = cv2.imread(os.path.join(self.img_path, img_name), 1)
        img_bgr_lr = cv2.resize(img_bgr_hr,
                                (img_bgr_hr.shape[1] // self.sr_factor, img_bgr_hr.shape[0] // self.sr_factor))

        img_bgr_lr, img_bgr_hr = img_bgr_lr.transpose((2, 0, 1)), img_bgr_hr.transpose((2, 0, 1))

        if self.normalzero2one:
            img_bgr_hr, img_bgr_lr = img_bgr_hr / 255., img_bgr_lr / 255.

        if self.has_name:
            return img_bgr_hr, img_bgr_lr, img_name
        else:
            return img_bgr_hr, img_bgr_lr

    def __len__(self):
        return len(self.img_list)


class SR_BGR_Data_Loader():
    def __init__(self, img_path, batch_size, normalzero2one=False, has_name=False, shuf=True, sr_factor=1):
        self.dataset = SR_BGR_Data(img_path, normalzero2one=normalzero2one, has_name=has_name, sr_factor=sr_factor)
        self.batch = batch_size
        self.shuf = shuf

    def loader(self):
        loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch, shuffle=self.shuf,
                                             num_workers=0, drop_last=True)
        return loader

# if __name__=="__main__":
#     train_loader = SR_BGR_Data_Loader(img_path='./data/train/label/', batch_size=20, normalzero2one=True, has_name=False, shuf=False, sr_factor=8).loader()
#     for step, (img_bgr_hr, img_bgr_lr) in enumerate(train_loader):
#         print(img_bgr_hr.shape)
#         print(img_bgr_lr.shape)
#         exit(0)
#
#     train_loader = Color_BGR_Data_Loader(img_path='./data/train/label/', batch_size=20, normalzero2one=True, has_name=False, shuf=False).loader()
#     for step, (img_bgr, img_gray) in enumerate(train_loader):
#         print(img_bgr.shape)
#         print(img_bgr.shape)
#         exit(0)
