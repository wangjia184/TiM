import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import torch
import math
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


# Semantic grouping (indices based on your list order):
# 0:5_o_Clock_Shadow, 1:Arched_Eyebrows, 2:Attractive, 3:Bags_Under_Eyes, 4:Bald, 
# 5:Bangs, 6:Big_Lips, 7:Big_Nose, 8:Black_Hair, 9:Blond_Hair, 10:Blurry, 11:Brown_Hair,
# 12:Bushy_Eyebrows, 13:Chubby, 14:Double_Chin, 15:Eyeglasses, 16:Goatee, 17:Gray_Hair,
# 18:Heavy_Makeup, 19:High_Cheekbones, 20:Male, 21:Mouth_Slightly_Open, 22:Mustache,
# 23:Narrow_Eyes, 24:No_Beard, 25:Oval_Face, 26:Pale_Skin, 27:Pointy_Nose,
# 28:Receding_Hairline, 29:Rosy_Cheeks, 30:Sideburns, 31:Smiling, 32:Straight_Hair,
# 33:Wavy_Hair, 34:Wearing_Earrings, 35:Wearing_Hat, 36:Wearing_Lipstick,
# 37:Wearing_Necklace, 38:Wearing_Necktie, 39:Young

GLOBAL_ATTR_IDX = torch.tensor([
    2,   # Attractive
    7,   # Big_Lips
    10,  # Blurry
    15,  # Eyeglasses
    20,  # Male
    31,  # Smiling
    36,  # Wearing_Lipstick
    39   # Young
], dtype=torch.long)  # 8 attributes

LOCAL_ATTR_IDX = torch.tensor([
    i for i in range(40) if i not in GLOBAL_ATTR_IDX
], dtype=torch.long)  # 32 attributes

class CelebAParquetDataset(Dataset):
    def __init__(self, parquet_files_pattern: str, output_size: int = 64):
        self.dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files_pattern},
            split="train"
        )
        self.global_idx = GLOBAL_ATTR_IDX
        self.local_idx = LOCAL_ATTR_IDX
        self.output_size = output_size
        print(f"[INFO] Loaded {len(self.dataset)} samples. "
              f"Global attrs: {len(self.global_idx)}, Local attrs: {len(self.local_idx)}. "
              f"Output size: {output_size}×{output_size}.")

    def __len__(self):
        return len(self.dataset)

    def _center_crop_square_from_landmarks(self, img: Image.Image, landmarks: torch.Tensor, scale_factor: float = 2.5) -> Image.Image:
        """
        Crop square using facial geometry:
        - Center: nose
        - Scale: distance between eye-midpoint and mouth-midpoint × scale_factor
        """
        # landmarks: [lefteye_x, lefteye_y, righteye_x, righteye_y,
        #             nose_x, nose_y,
        #             leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y]
        le_x, le_y = landmarks[0].item(), landmarks[1].item()
        re_x, re_y = landmarks[2].item(), landmarks[3].item()
        no_x, no_y = landmarks[4].item(), landmarks[5].item()
        lm_x, lm_y = landmarks[6].item(), landmarks[7].item()
        rm_x, rm_y = landmarks[8].item(), landmarks[9].item()

        # Midpoint of eyes
        A_x = (le_x + re_x) / 2.0
        A_y = (le_y + re_y) / 2.0
        # Midpoint of mouth corners
        B_x = (lm_x + rm_x) / 2.0
        B_y = (lm_y + rm_y) / 2.0

        # Distance A–B
        d = math.hypot(B_x - A_x, B_y - A_y)
        if d < 1e-3:
            d = 50.0  # fallback for degenerate landmarks

        S = int(round(scale_factor * d))
        if S < 16:
            S = 16  # minimal reasonable face size

        W, H = img.size

        # If image too small, cap S
        if S > W or S > H:
            S = min(W, H)

        # Initial center = nose
        cx0, cy0 = no_x, no_y

        # Compute valid top-left range for an S×S crop: [0, W-S] × [0, H-S]
        left_min, left_max = 0, max(0, W - S)
        top_min,  top_max  = 0, max(0, H - S)

        # Ideal top-left if centered at (cx0, cy0)
        left_ideal = cx0 - S // 2
        top_ideal  = cy0 - S // 2

        # Shift minimally to stay in bounds
        left = int(round(max(left_min, min(left_ideal, left_max))))
        top  = int(round(max(top_min,  min(top_ideal,  top_max))))

        right = left + S
        bottom = top + S

        # Final safety clamp (should be redundant)
        left   = max(0, left)
        top    = max(0, top)
        right  = min(W, right)
        bottom = min(H, bottom)
        if right <= left: right = left + 1
        if bottom <= top:  bottom = top + 1

        return img.crop((left, top, right, bottom))

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img_raw = sample["image"]
        landmarks = torch.tensor(sample["landmarks"], dtype=torch.int32)
        img_cropped = self._center_crop_square_from_landmarks(img_raw, landmarks, scale_factor=3.0)
        img_resized = img_cropped.resize((self.output_size, self.output_size), Image.BILINEAR)
        # TiM reference code trains in [-1, 1] space (before VAE). We keep that convention in pixel-space.
        img_tensor = F.to_tensor(img_resized) * 2.0 - 1.0  # [3, H, W] in [-1, 1]

        # Get full attributes as float32
        all_attrs = torch.tensor(sample["attributes"], dtype=torch.float32)  # [40]

        # Split into global and local
        global_attrs = all_attrs[self.global_idx]   # [8]
        local_attrs = all_attrs[self.local_idx]     # [32]

        return {
            "image": img_tensor,        # [3, 64, 64]
            "global_attrs": global_attrs,  # [8]
            "local_attrs": local_attrs,    # [32]
        }

def save_sample_images(dataset: CelebAParquetDataset, indices=[0, 10, 100], output_dir="sample_images"):
    os.makedirs(output_dir, exist_ok=True) 

    for idx in indices:
        if idx >= len(dataset):
            print(f"[WARN] Index {idx} out of range (N={len(dataset)}), skipping.")
            continue
        item = dataset[idx] 
        img_tensor = item["image"]  # [3, H, W]
 

        print(f"\n--- Sample #{idx} ---") 
        print(f" Global Attributes: {item['global_attrs'].shape}") # [8]
        print(f" Local Attributes: {item['local_attrs'].shape}")  # [32]

        
        # Save images
        base = Path(output_dir) / f"sample_{idx:04d}"   
        final_pil = F.to_pil_image(img_tensor)
        final_pil.save(base.with_suffix(".64x64.jpg"))
        print(f"  Saved:  .final_64x64.jpg")

if __name__ == "__main__":
    parquet_pattern = "../celeba/data/train-*.parquet"
    dataset = CelebAParquetDataset(parquet_pattern, output_size=64)
    save_sample_images(dataset, indices=[1, 2, 3, 4, 5, 6, 11, 111])