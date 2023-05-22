import torch
import torchaudio
import matplotlib.pyplot as plt
# import librosa
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from os import listdir
from os.path import isfile, join

def load_data(datapath, metadata_path, transform=torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_mels=16)):
    df = pd.read_csv(metadata_path, sep='\t')
    features = torch.empty(size=[1, 16, 887])
    labels = torch.empty(size=[1])
    for idx in range(len(df)):
        audio_file = datapath + df.loc[idx, 'filename']
        label = df.loc[idx, 'label']
        waveform, sample_rate = torchaudio.load(audio_file)
        specgram = transform(waveform)
        features = torch.cat((features, specgram), dim=0)
        labels = torch.cat((labels, torch.tensor([label])), dim=0)
    features = features[1:,:,:] # remove first empty tensor
    labels = labels[1:] # remove first empty tensor
    data = TensorDataset(features, labels)
    return data

def write_metadata(train_path, test_path):
    train_files = sorted([f for f in listdir(train_path) if isfile(join(train_path, f))])
    train_label = [int(name[0].islower()) for name in train_files]

    test_files = sorted([f for f in listdir(test_path) if isfile(join(test_path, f))])
    test_label = [int(name[0].islower()) for name in test_files]

    train_df = pd.DataFrame({'filename': train_files,
                    'label': train_label})
    test_df = pd.DataFrame({'filename': test_files,
                    'label': test_label})


if __name__=="__main__":
    # train_data = torch.load("train_data.pt")
    # val_data = torch.load("val_data.pt")
    # test_data = torch.load("test_data.pt")
    # print(len(train_data))
    # print(len(val_data))
    # print(len(test_data))
    # print(test_data[0])

    metadata_train = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/metadata_train.csv"
    train_path = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/train_data/"

    metadata_test = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/metadata_test.csv"
    test_path = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/test_data/"

    train_data = load_data(train_path, metadata_train)
    print("finish loading train data")
    test_data = load_data(test_path, metadata_test)
    print("finish loading test data")

    # Random split of 80:20 between training and validation
    torch.manual_seed(0)
    train_len = len(train_data)
    num_train = round(train_len * 0.8)
    num_val = train_len - num_train
    train_ds, val_ds = random_split(train_data, [num_train, num_val])
    print(len(train_ds), len(val_ds))
    # save data
    save_data = True
    if save_data:
        torch.save(train_ds, "train_data.pt")
        torch.save(val_ds, "val_data.pt")
        torch.save(test_data, "test_data.pt")
        print("finish saving data")
    
# class SoundDS(Dataset):
#     def __init__(self, df, data_path, transform=torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_mels=16)):
#         self.df = df
#         self.data_path = str(data_path)
#         self.duration = 4000
#         self.sr = 44100
#         self.channel = 1
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.df) 
    
#     def __getitem__(self, idx):
#         audio_file = self.data_path + self.df.loc[idx, 'filename']
#         label = self.df.loc[idx, 'label']
#         waveform, sample_rate = torchaudio.load(audio_file)
#         specgram = self.transform(waveform)
#         return (specgram, label)