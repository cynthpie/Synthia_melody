import torch
import torchaudio
import matplotlib.pyplot as plt
# import librosa
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed

def load_data(datapath, metadata_path, transform=torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=32)):
    metadata_df = pd.read_csv(metadata_path, sep=',')
    output = Parallel(n_jobs=-1)(delayed(data_transform)(metadata_df.iloc(0)[i], j) for (i, j) in zip(range(len(metadata_df)), [datapath]*len(metadata_df)))
    features = torch.stack([tensor_tuple[0] for tensor_tuple in output])
    labels = torch.stack([tensor_tuple[1] for tensor_tuple in output])
    data = TensorDataset(features, labels)
    return data

def data_transform(one_metadata,datapath):
    df = one_metadata
    d = {"major":1, "minor":0}
#    datapath = "/rds/general/user/cl222/home/audio/unbiased_train/"
    audio_file = datapath + one_metadata['filename']+ ".wav"
    label = one_metadata['scale']
    waveform, sample_rate = torchaudio.load(audio_file)
    transform=torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=32)
    specgram = transform(waveform)
    return (specgram, torch.tensor([d[label]]))


if __name__=="__main__":
    # set data path 
    sine_train_path = "/rds/general/user/cl222/home/audio/sine_train"
    metadata_sine_train = "/rds/general/user/cl222/home/audio/metadata_sine_train.csv"
    sine_test_path = "/rds/general/user/cl222/home/audio/sine_test"
    metadata_sine_test = "/rds/general/user/cl222/home/audio/metadata_sine_test.csv"

    square_train_path = "/rds/general/user/cl222/home/audio/square_train"
    metadata_square_train = "/rds/general/user/cl222/home/audio/metadata_square_train.csv"
    square_test_path = "/rds/general/user/cl222/home/audio/square_test"
    metadata_square_test = "/rds/general/user/cl222/home/audio/metadata_square_test.csv"

    sawtooth_train_path = "/rds/general/user/cl222/home/audio/sawtooth_train"
    metadata_sawtooth_train = "/rds/general/user/cl222/home/audio/metadata_sawtooth_train.csv"
    sawtooth_test_path = "/rds/general/user/cl222/home/audio/sawtooth_test"
    metadata_sawtooth_test = "/rds/general/user/cl222/home/audio/metadata_sawtooth_test.csv"

    triangle_train_path = "/rds/general/user/cl222/home/audio/triangle_train"
    metadata_triangle_train = "/rds/general/user/cl222/home/audio/metadata_triangle_train.csv"
    triangle_test_path = "/rds/general/user/cl222/home/audio/triangle_test"
    metadata_triangle_test = "/rds/general/user/cl222/home/audio/metadata_triangle_test.csv"

    # load data
    sine_train_data = load_data(sine_train_path, metadata_sine_train)
    sine_test_data = load_data(sine_test_path, metadata_sine_test)
    print("finish loading sine data")
    square_train_data = load_data(square_train_path, metadata_square_train)
    square_test_data = load_data(square_test_path, metadata_square_test)
    print("finish loading square data")
    sawtooth_train_data = load_data(sawtooth_train_path, metadata_sawtooth_train)
    sawtooth_train_data = load_data(sawtooth_test_path, metadata_sawtooth_test)
    print("finish loading sawtooth data")
    triangle_train_data = load_data(triangle_train_path, metadata_triangle_train)
    triangle_test_data = load_data(triangle_test_path, metadata_triangle_test)
    print("finish loading triangle data")

    # Random split of 80:20 between training and validation
    torch.manual_seed(0)
    train_len = len(train_data)
    num_train = round(train_len * 0.8)
    num_val = train_len - num_train
    sine_train_ds, sine_val_ds = random_split(train_data, [num_train, num_val])
    print(len(train_ds), len(val_ds))
    # save data
    save_data = True
    if save_data:
        torch.save(train_ds, "/rds/general/user/cl222/home/msc_project/classification/biased_train_data.pt")
        torch.save(val_ds, "/rds/general/user/cl222/home/msc_project/classification/biased_val_data.pt")
        torch.save(test_data, "/rds/general/user/cl222/home/msc_project/classification/anti-biased_test_data.pt")
        torch.save(neutral_data, "/rds/general/user/cl222/home/msc_project/classification/neutral_test_data.pt")
        print("finish saving data")

    train_data = torch.load("/rds/general/user/cl222/home/msc_project/classification/biased_train_data.pt")
    val_data = torch.load("/rds/general/user/cl222/home/msc_project/classification/biased_val_data.pt")
    test_data = torch.load("/rds/general/user/cl222/home/msc_project/classification/anti-biased_test_data.pt")
    neutral_data = torch.load("/rds/general/user/cl222/home/msc_project/classification/neutral_test_data.pt")
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    print(len(neutral_data))
    