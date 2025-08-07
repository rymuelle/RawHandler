import random
from torch.utils.data import Dataset

from RawHandler.RawHandler import RawHandler
from RawHandler.utils import align_images


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
        H_center = random.randint(0 + half_crop, H - half_crop)
        W_center = random.randint(0 + half_crop, W - half_crop)
        crop = (
            H_center - half_crop,
            H_center + half_crop,
            W_center - half_crop,
            W_center + half_crop,
        )
        if self.offsets is None:
            offset = (0, 0, 0, 0)
        else:
            offset = self.offsets[idx]
        offset = align_images(noisy_rh, gt_rh, crop, offset=(0, 0, 0, 0))

        # Adjust exposure
        noisy_rggb = noisy_rh.as_rggb(dims=crop)
        noisy_rgb = noisy_rh.as_rgb(dims=crop)
        clean_rgb = gt_rh.as_rgb(dims=crop)

        if self.transform:
            noisy_rggb = self.transform(noisy_rggb)
            noisy_rgb = self.transform(noisy_rgb)
            clean_rgb = self.transform(clean_rgb)
        return noisy_rggb, noisy_rgb, clean_rgb, offset
