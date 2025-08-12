import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from evaluate import load

import torch
from torchvision import transforms
from transformers import AutoImageProcessor
from sentence_transformers import SentenceTransformer

from dataset import FungiDataset


def embed_string(x: pd.Series) -> pd.Series:
    unique_tokens = x.unique()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_tokens)
    embedding_map = dict(zip(unique_tokens, embeddings))
    return x.map(embedding_map)

def encode_datetime(x: pd.Series) -> tuple:
    x = pd.to_datetime(x)
    x_year = x.dt.year
    day_of_year = x.dt.dayofyear
    angle = 2 * np.pi * (day_of_year - 1) / 365.25
    return x_year, np.sin(angle), np.cos(angle)


def collate_fn(batch):
    images, labels, file_paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"pixel_values": images, "labels": labels}


def get_transforms(split):
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])
    else:
        return transforms.Compose([
        transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])

def get_dataloader(df, path):
    train_df = df[df['filename_index'].str.contains('train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print('+++++++++++++++++++++++++++++++')
    print('VAL SAMPLES: ', len(val_df))
    print('TRAINING SAMPLES: ', len(train_df))
    test_df = df[df['filename_index'].str.contains('test')]

    train_dataset = FungiDataset(df=train_df, path=path, transform=get_transforms('train'))
    val_dataset = FungiDataset(df=val_df, path=path, transform=get_transforms('val'))
    test_dataset = FungiDataset(df=test_df, path=path, transform=get_transforms('test'))
    return train_dataset, val_dataset, test_dataset

metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)