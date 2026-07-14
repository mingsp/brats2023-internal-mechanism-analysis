import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir

        split_candidates: Dict[str, List[Tuple[str, str]]] = {
            "train": [
                ("train_Image", "train_Mask"),
                ("train_image", "train_Mask"),
                ("train_image", "train_mask"),
                ("train_Image", "train_mask"),
            ],
            "val": [
                ("val_Image", "val_Mask"),
                ("val_image", "val_Mask"),
                ("val_image", "val_mask"),
                ("val_Image", "val_mask"),
            ],
            "test": [
                ("test_Image", "test_Mask"),
                ("test_image", "test_Mask"),
                ("test_image", "test_mask"),
                ("test_Image", "test_mask"),
            ],
        }

        if self.split not in split_candidates:
            raise ValueError(
                f"Unsupported split: {self.split}. Expected one of {tuple(split_candidates)}"
            )

        self.image_dir, self.mask_dir = self._resolve_split_dirs(base_dir, split_candidates[self.split])

        self.sample_list = [f.replace(".npy", "") for f in os.listdir(self.image_dir) if f.endswith(".npy")]
        self.sample_list.sort()

        print(f"[{split}] Loaded {len(self.sample_list)} samples from {self.image_dir}")

    @staticmethod
    def _resolve_split_dirs(base_dir: str, candidates: List[Tuple[str, str]]) -> Tuple[str, str]:
        tried = []
        for image_subdir, mask_subdir in candidates:
            image_dir = os.path.join(base_dir, image_subdir)
            mask_dir = os.path.join(base_dir, mask_subdir)
            tried.append((image_dir, mask_dir))
            if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
                return image_dir, mask_dir
        hint = "\n".join([f"  - {img} | {msk}" for img, msk in tried])
        raise FileNotFoundError(
            "Could not resolve split directories under {}. Tried:\n{}".format(base_dir, hint)
        )

    @staticmethod
    def _ensure_chw_image(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape={image.shape}")
        if image.shape[0] == 4:
            return image
        if image.shape[-1] == 4:
            return image.transpose(2, 0, 1)
        raise ValueError(f"Expected 4-channel image, got shape={image.shape}")

    @staticmethod
    def _ensure_hw_label(label: np.ndarray) -> np.ndarray:
        if label.ndim == 2:
            return label

        if label.ndim == 3:
            if label.shape[0] in (1, 3, 4):
                label_map = np.argmax(label, axis=0)
                return label_map.astype(np.uint8)
            if label.shape[-1] in (1, 3, 4):
                label_map = np.argmax(label, axis=-1)
                return label_map.astype(np.uint8)

        raise ValueError(f"Unsupported label shape={label.shape}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        img_path = os.path.join(self.image_dir, slice_name + ".npy")
        mask_path = os.path.join(self.mask_dir, slice_name + ".npy")

        image = np.load(img_path).astype(np.float32)
        label = np.load(mask_path)

        image = self._ensure_chw_image(image)
        label = self._ensure_hw_label(label)

        label = label.astype(np.uint8)
        label[label == 4] = 3

        sample = {"image": torch.from_numpy(image), "label": torch.from_numpy(label).long()}
        sample["case_name"] = slice_name
        return sample
