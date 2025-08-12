import os.path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]
        label = self.df['taxonID_index'].values[idx]  # Get label
        if pd.isnull(label):
            label = -1
        else:
            label = int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, file_path
