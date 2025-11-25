import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A

POLYP_DATA_PATH = '/content/drive/MyDrive/polyp_segment/data_polyp/polypData.npz'
PRETRAINED_PATH = '/content/drive/MyDrive/polyp_segment/Weight/pvt_v2_b5.pth'

# Argumentation
train_transform = A.Compose([
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


# Polyp Dataset
class PolypDS(Dataset):
    def __init__(self, data_path=POLYP_DATA_PATH, type=None, transform = None):
      super().__init__()

      data_np = np.load(data_path)
      self.images = data_np[f"{type}_img"]
      self.masks  = data_np[f"{type}_msk"].squeeze(-1)
      self.transform = transform

    def __getitem__(self, idx):
      img = self.images[idx]
      msk  = self.masks[idx]

      if self.transform is not None:
            transformed = self.transform(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]

      img = transforms.ToTensor()(img)
      msk = np.expand_dims(msk, axis = -1)
      msk = transforms.ToTensor()(msk)

      return img, msk

    def __len__(self):
      return len(self.images)

# Dataset
train_ds = PolypDS(type = 'train', transform=train_transform)
val_ds   = PolypDS(type = 'val', transform=val_transform)
test1_ds = PolypDS(type = 'test_kvasir', transform=val_transform)
test2_ds = PolypDS(type = 'test_etis', transform=val_transform)
test3_ds = PolypDS(type = 'test_cvc300', transform=val_transform)
test4_ds = PolypDS(type = 'test_clinic', transform=val_transform)
test5_ds = PolypDS(type = 'test_colon', transform=val_transform)


# DataLoader
trainloader = DataLoader(train_ds, batch_size=16, num_workers=2, shuffle=True)
valloader   = DataLoader(val_ds, batch_size=8, num_workers=2, shuffle=False)
testloader1 = DataLoader(test1_ds, batch_size=1, num_workers=2, shuffle=False)
testloader2 = DataLoader(test2_ds, batch_size=1, num_workers=2, shuffle=False)
testloader3 = DataLoader(test3_ds, batch_size=1, num_workers=2, shuffle=False)
testloader4 = DataLoader(test4_ds, batch_size=1, num_workers=2, shuffle=False)
testloader5 = DataLoader(test5_ds, batch_size=1, num_workers=2, shuffle=False)