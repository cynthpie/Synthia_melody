print("beginning of script")
import torch
import torchaudio
import matplotlib.pyplot as plt
# import librosa
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import os, sys
from os import listdir
from os.path import isfile, join
import pandas as pd
sys.path.append('/rds/general/user/cl222/home/msc_project/classification')
from dataset import CynthiaDataset
from torchaudio.transforms import MelSpectrogram
from joblib import Parallel, delayed
print("finish import packaget")

sine_train_path = "/rds/general/user/cl222/home/audio/sine_train/"
metadata_sine_train = "/rds/general/user/cl222/home/audio/metadata_sine_train.csv"
sine_test_path = "/rds/general/user/cl222/home/audio/sine_test/"
metadata_sine_test = "/rds/general/user/cl222/home/audio/metadata_sine_test.csv"

square_train_path = "/rds/general/user/cl222/home/audio/square_train/"
metadata_square_train = "/rds/general/user/cl222/home/audio/metadata_square_train.csv"
square_test_path = "/rds/general/user/cl222/home/audio/square_test/"
metadata_square_test = "/rds/general/user/cl222/home/audio/metadata_square_test.csv"

sawtooth_train_path = "/rds/general/user/cl222/home/audio/sawtooth_train/"
metadata_sawtooth_train = "/rds/general/user/cl222/home/audio/metadata_sawtooth_train.csv"
sawtooth_test_path = "/rds/general/user/cl222/home/audio/sawtooth_test/"
metadata_sawtooth_test = "/rds/general/user/cl222/home/audio/metadata_sawtooth_test.csv"

triangle_train_path = "/rds/general/user/cl222/home/audio/triangle_train/"
metadata_triangle_train = "/rds/general/user/cl222/home/audio/metadata_triangle_train.csv"
triangle_test_path = "/rds/general/user/cl222/home/audio/triangle_test/"
metadata_triangle_test = "/rds/general/user/cl222/home/audio/metadata_triangle_test.csv"

def preprocess_data(data_loader):
    data = []
    for idx, (x, y) in enumerate(data_loader):
        data.append((x, y))
    features = torch.concat([tensor_tuple[0] for tensor_tuple in data])
    labels = torch.concat([tensor_tuple[1] for tensor_tuple in data])
    data = TensorDataset(features, labels)
    return data

if __name__=="__main__":

    sine_train_data = CynthiaDataset(metadata_sine_train, sine_train_path)
    square_train_data = CynthiaDataset(metadata_square_train, square_train_path)
    sawtooth_train_data = CynthiaDataset(metadata_sawtooth_train, sawtooth_train_path)
    triangle_train_data = CynthiaDataset(metadata_triangle_train, triangle_train_path)

    # # train-val split
    train_len = len(square_train_data)
    num_train = round(train_len * 0.8)
    num_val = train_len - num_train
    sine_train_ds, sine_val_ds = random_split(sine_train_data, [num_train, num_val], torch.Generator().manual_seed(42))
    square_train_ds, square_val_ds = random_split(square_train_data, [num_train, num_val], torch.Generator().manual_seed(42))
    sawtooth_train_ds, sawtooth_val_ds = random_split(sawtooth_train_data, [num_train, num_val], torch.Generator().manual_seed(42))
    triangle_train_ds, triangle_val_ds = random_split(triangle_train_data, [num_train, num_val], torch.Generator().manual_seed(42))

    sine_train_loader = DataLoader(sine_train_ds, batch_size=len(sine_train_ds)//200, num_workers=200)
    sine_val_loader = DataLoader(sine_val_ds, batch_size=len(sine_val_ds)//200, num_workers=200)
    square_train_loader = DataLoader(square_train_ds, batch_size=len(square_train_ds)//200, num_workers=200)
    square_val_loader = DataLoader(square_val_ds, batch_size=len(square_val_ds)//200, num_workers=200)
    sawtooth_train_loader = DataLoader(sawtooth_train_ds, batch_size=len(sawtooth_train_ds)//200, num_workers=200)
    sawtooth_val_loader = DataLoader(sawtooth_val_ds, batch_size=len(sawtooth_val_ds)//200, num_workers=200)
    triangle_train_loader = DataLoader(triangle_train_ds, batch_size=len(triangle_train_ds)//200, num_workers=200)
    triangle_val_loader = DataLoader(triangle_val_ds, batch_size=len(triangle_val_ds)//200, num_workers=200)

    sine_train_ds = preprocess_data(sine_train_loader)
    sine_val_ds = preprocess_data(sine_val_loader)
    square_train_ds = preprocess_data(square_train_loader)
    square_val_ds = preprocess_data(square_val_loader)
    sawtooth_train_ds = preprocess_data(sawtooth_train_loader)
    sawtooth_val_ds = preprocess_data(sawtooth_val_loader)
    triangle_train_ds = preprocess_data(triangle_train_loader)
    triangle_val_ds = preprocess_data(triangle_val_loader)
    print("finish preprocess train/val data")

    torch.save(sine_train_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/sine_train_data.pt")
    torch.save(sine_val_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/sine_val_data.pt")
    torch.save(square_train_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/square_train_data.pt")
    torch.save(square_val_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/square_val_data.pt")
    torch.save(sawtooth_train_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/sawtooth_train_data.pt")
    torch.save(sawtooth_val_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/sawtooth_val_data.pt")
    torch.save(triangle_train_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/triangle_train_data.pt")
    torch.save(triangle_val_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/triangle_val_data.pt")

    data = torch.load("square_train_data.pt")
    print(len(data))

    # test data
    sine_test_ds = CynthiaDataset(metadata_sine_test, sine_test_path)
    square_test_ds = CynthiaDataset(metadata_square_test, square_test_path)
    sawtooth_test_ds = CynthiaDataset(metadata_sawtooth_test, sawtooth_test_path)
    triangle_test_ds = CynthiaDataset(metadata_triangle_test, triangle_test_path)

    sine_test_loader = DataLoader(sine_test_ds,  batch_size=len(sine_test_ds)//200, num_workers=200)
    square_test_loader = DataLoader(square_test_ds,  batch_size=len(square_test_ds)//200, num_workers=200)
    sawtooth_test_loader = DataLoader(sawtooth_test_ds,  batch_size=len(sawtooth_test_ds)//200, num_workers=200)
    triangle_test_loader = DataLoader(triangle_test_ds,  batch_size=len(triangle_test_ds)//200, num_workers=200)

    sine_test_ds = preprocess_data(sine_test_loader)
    square_test_ds = preprocess_data(square_test_loader)
    sawtooth_test_ds = preprocess_data(sawtooth_test_loader)
    triangle_test_ds = preprocess_data(triangle_test_loader)

    torch.save(sine_test_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/sine_test_data.pt")
    torch.save(square_test_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/square_test_data.pt")
    torch.save(sawtooth_test_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/sawtooth_test_data.pt")
    torch.save(triangle_test_ds, "/rds/general/user/cl222/home/msc_project/sampleCNN/triangle_test_data.pt")

    data = torch.load("sine_test_data.pt")
    print(len(data))
