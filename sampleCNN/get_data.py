import torch
import torchaudio
import os, sys
sys.path.append('/rds/general/user/cl222/home/msc_project/classification')
from dataset import CynthiaDataset
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset, Subset, ConcatDataset
import pandas as pd

def get_dataset(data_dir, usage, shift_type=None, waveshape=None, shift_strength=0, shift_way=None):
    """
    Function to build dataset shift scenario from preprocessed data
    Args:
        - data_dir (str): data directory to store preprocessed data
        - waveshape (str): waveshape made up of the data if no shift. Can be "sine", "square", "triangle", "sawtooth"
        - usage (str): usage of data. Can be "train", "val", "test".
        - shift_type (str): type of dataset shift. Can be "domain_shift", "sample_selection_bias", or None
        - shift_strength (float): strength of dataset shift in [0.0, 1.0]. Proportion of shifted sample in training and val data 
        - shift_way (list): way to construct shift. 
            domain_shift in the form: [orig_shape, shift_shape]. E.g. ["sine", "square"]
            sample_selecti_bias in the form: [orig_shape, shift_shape]. Major associated with orig_shape. E.g. ["sine", "square"]
            If no shift, shift_way = None
    Return:
        - dataset (torch.utils.data.Subset): train dataset with specified shift
    """
    # check input
    assert shift_type in ["domain_shift", "sample_selection_bias", None]
    if shift_type is None:
        assert waveshape is not None
        shift_strength = 0
        shift_way = None
    elif shift_type == "domain_shift":
        assert shift_way is not None
    elif shift_type == "sample_selection_bias":
        assert shift_way is not None
    assert shift_strength >= 0.0 and shift_strength <= 1.0

    # load data
    if shift_type is None:
        dataset = torch.load(data_dir + f"/{waveshape}_{usage}_ds.pt")
        data_len = len(dataset)
    else:
        dataset1 = torch.load(data_dir + f"/{shift_way[0]}_{usage}_ds.pt")
        dataset2 = torch.load(data_dir + f"/{shift_way[1]}_{usage}_ds.pt")
        data_len = len(dataset1)

    # extract data
    if shift_type is None:
        return dataset
    # index 1 to 20000 in train is major, 20001 to 40000 is minor
    elif shift_type == "domain_shift":
        minor_start_index = data_len//2
        orig_major_indicies = list(range(int(minor_start_index*(1-shift_strength))))
        orig_minor_indicies = list(range(minor_start_index, int(minor_start_index*(1+1-shift_strength))))
        orig_indicies = orig_major_indicies + orig_minor_indicies
        shift_indicies = [i for i in range(data_len) if i not in orig_indicies]
        orig_dataset = Subset(dataset1, orig_indicies)
        shift_dataset = Subset(dataset2, shift_indicies)
        dataset = ConcatDataset([orig_dataset, shift_dataset])
        return dataset
        
    elif shift_type == "sample_selection_bias":
        # if bias_strength==0, let 1 to 10000, 20001 to 30000 be orig, 10001 to 20000, 30001 to 40000 be shift
        # if bias, orig give 20001 to 30000 away, shift give 10001 to 20000 away
        minor_start_index = data_len//2
        orig_minor_indicies = list(range(minor_start_index, minor_start_index + data_len//4))
        shift_major_indicies = list(range(data_len//4, minor_start_index))
        nb_bias_sample = int((data_len//4) * shift_strength)
        give_away_to_shift = orig_minor_indicies[0:nb_bias_sample]
        orig_minior_left = orig_minor_indicies[nb_bias_sample:]
        give_away_to_orig = shift_major_indicies[0:nb_bias_sample]
        shift_major_left =  shift_major_indicies[nb_bias_sample:]
        orig_indicies = list(range(data_len//4)) + orig_minior_left + give_away_to_orig
        shift_indicies = list(range(data_len-data_len//4, data_len)) + shift_major_left + give_away_to_shift
        orig_dataset = Subset(dataset1, orig_indicies)
        shift_dataset = Subset(dataset2, shift_indicies)
        dataset = ConcatDataset([orig_dataset, shift_dataset])
        return dataset
    return None

if __name__ == "__main__":
    data = get_dataset(data_dir="", usage="val", shift_type="sample_selection_bias", shift_strength=1, shift_way=["sine", "square"])
    print(data, len(data))