import torch
import torchaudio
import matplotlib.pyplot as plt
# import librosa
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, random_split

class SoundDS(Dataset):
    def __init__(self, df, data_path, transform=torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_mels=16)):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 1
        self.transform = transform
        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'filename']
        label = self.df.loc[idx, 'label']
        waveform, sample_rate = torchaudio.load(audio_file)
        specgram = self.transform(waveform)
        return (specgram, label)

if __name__=="__main__":
    metadata_train = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/metadata_train.csv"
    train_path = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/train_data/"
    train_meta = pd.read_csv(metadata_train, sep='\t')
    train_ds = SoundDS(train_meta, train_path)

    metadata_test = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/metadata_test.csv"
    test_path = "/Users/chia-hsinlin/Documents/Imperial/individual_project/msc_project/tempsnd/test_data/"
    test_meta = pd.read_csv(metadata_test, sep='\t')
    test_ds = SoundDS(test_meta, test_path)

    # Random split of 80:20 between training and validation
    train_len = len(train_ds)
    num_train = round(train_len * 0.8)
    num_val = train_len - num_train
    train_ds, val_ds = random_split(train_ds, [num_train, num_val])


    # save data
    save_data = False
    if save_data:
        torch.save(train_ds, "train_data.pt")
        torch.save(val_ds, "val_data.pt")
        torch.save(test_ds, "test_data.pt")
    
