import random
from torch.utils.data import Dataset
import torch

from RawHandler.RawHandler import RawHandler

import re


class RawDataset(Dataset):
    def __init__(self, file_pair_list, transform=None, crop_size=256, offsets=None):
        super().__init__()
        self.file_pair_list = file_pair_list
        self.transform = transform
        self.crop_size = crop_size
        self.offsets = offsets

    def __len__(self):
        return len(self.file_pair_list)

    def __getitem__(self, idx):
        noisy_file = self.file_pair_list[idx][0]
        gt_file = self.file_pair_list[idx][1]
        noisy_rh = RawHandler(noisy_file)
        gt_rh = RawHandler(gt_file)

        # Crop and align
        H, W = noisy_rh.raw.shape[-2:]
        half_crop = self.crop_size // 2
        H_center = random.randint(0 + half_crop * 2, H - half_crop * 2)
        W_center = random.randint(0 + half_crop * 2, W - half_crop * 2)
        crop = (
            H_center - half_crop,
            H_center + half_crop,
            W_center - half_crop,
            W_center + half_crop,
        )
        if self.offsets is None:
            offset = (0, 0, 0, 0)
        else:
            offset = self.offsets[idx][0][0]

        # Adjust exposure
        gain = (
            noisy_rh.adjust_bayer_bw_levels(dims=crop).mean()
            / gt_rh.adjust_bayer_bw_levels(dims=crop).mean()
        )
        gt_rh.gain = gain

        # offset = align_images(noisy_rh, gt_rh, crop, offset=offset, step_sizes=[2])

        noisy_rggb = noisy_rh.as_rggb_colorspace(dims=crop, colorspace="AdobeRGB")
        noisy_rgb = noisy_rh.as_rgb_colorspace(dims=crop, colorspace="AdobeRGB")
        clean_rgb = gt_rh.as_rgb_colorspace(dims=crop + offset, colorspace="AdobeRGB")

        iso = re.findall("_ISO([0-9]+)_", noisy_file)
        if len(iso) == 1:
            iso = int(iso[0])
        else:
            iso = -100

        iso_conditioning = iso / 65535

        if self.transform:
            noisy_rggb = self.transform(noisy_rggb.transpose(1, 2, 0))
            noisy_rgb = self.transform(noisy_rgb.transpose(1, 2, 0))
            clean_rgb = self.transform(clean_rgb.transpose(1, 2, 0))
            iso_conditioning = torch.tensor([iso_conditioning])

        return noisy_rggb, noisy_rgb, clean_rgb, offset, iso_conditioning
