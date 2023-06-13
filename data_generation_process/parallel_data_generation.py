# data generation process
import numpy as np
import copy
import random
import pandas as pd
from scipy.io import wavfile
import librosa
import sys
import os
sys.path.append('/rds/general/user/cl222/home/msc_project/')
from synth.components.composers import WaveAdder
from synth.components.oscillators.modulated_oscillator import ModulatedOscillator
from synth.components.oscillators.oscillators import SineOscillator, SquareOscillator, SawtoothOscillator, TriangleOscillator
from synth.components.envelopes import ADSREnvelope
from synth.components.freqencymod import FrequencyModulator

SR = 16000 # sample rate #16000
MAJOR_FREQ = {
    "C_":261.6256, 
    "Db":277.1826, 
    "D_":293.6648, 
    "Eb":311.1270, 
    "E_":329.6276, 
    "F_":349.2282, 
    "Gb":369.9944, 
    "G_":391.9954, 
    "Ab":415.3047,
    "A_":440.0000, 
    "Bb":466.1638, 
    "B_":493.8833}

MINOR_FREQ = {
    "c_":261.6256, 
    "c#":277.1826, 
    "d_":293.6648, 
    "eb":311.1270, 
    "e_":329.6276, 
    "f_":349.2282, 
    "f#":369.9944, 
    "g_":391.9954, 
    "g#":415.3047,
    "a_":440.0000, 
    "bb":466.1638, 
    "b_":493.8833}

#####################################################################
# Helper functions
hz = lambda note:librosa.note_to_hz(note)
to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))

def wave_to_file(wav, wav2=None, fname="temp", amp=0.1, scale_type="major"):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    if scale_type == "major":
        try:
            wavfile.write(f"../../audio/major_24/{fname}.wav", SR, wav)
        except FileNotFoundError:
            os.makedirs('../../audio/major_24/')
            wavfile.write(f"../../audio/major_24/{fname}.wav", SR, wav)
    elif scale_type == "minor": 
        try:
            wavfile.write(f"../../audio/minor_24/{fname}.wav", SR, wav)
        except FileNotFoundError:
            os.makedirs('../../audio/minor_24/')
            wavfile.write(f"../../audio/minor_24/{fname}.wav", SR, wav)

def amp_mod(init_amp, env):
    return env * init_amp
def freq_mod(init_freq, env, mod_amt=0.01, sustain_level=0.7):
    return init_freq + ((env - sustain_level) * init_freq * mod_amt)
def freq_mod_2(init_freq, env):
    return env

def getval(osc, count=SR, it=False):
    if it: osc = iter(osc)
    # returns 1 sec of samples of given osc.
    return [next(osc) for i in range(count)]

def gettrig(gen, downtime, sample_rate=SR):
    gen = iter(gen)
    down = int(downtime * sample_rate)
    vals = getval(gen, down)
    gen.trigger_release()
    while not gen.ended:
        vals.append(next(gen))
    return vals

def get_octive(freq, freq_range):
    octives = [2, 4]
    octive_list = []
    octive_up = [freq*e for e in octives]
    octive_down = [freq/e for e in octives]
    octive_list = [freq] + octive_up + octive_down
    octive_list = [e for e in octive_list if (e > freq_range[0] and e < freq_range[1])]
    return octive_list

##########################################################################
# generate metadata file
def generate_metadata_file(nb_sample, major_prop, bias, bias_type, bias_strength, controlling_factors, noise_level, data_use):
    """ 
    Args:  
        - nb_sample (int): number of samples to generate 
        - major_prop (float): percentage of major samples in nb_sample. Range: [0.0, 1.0]
        - bias (str): bias to correlate with label and signal: "wave_shape", "amplitude", or "freq_range"
        - bias_type (dict): how bias correlate with label, e.g. {major:sine, minor:square}
        - bias_strength (float): strength of bias correlation with majors. Range: [0.0, 1.0] 
        - controlling_factors (dict): dic of {factor : characteristic}. e.g. {"wave_shape" : "sine"}
        - noise_level (float): noise level in the dataset. Range: [0.0, 1.0] 
        - data_use (str): "train" or "test"

    Return:
        - metadata_csv (csv): csv containing characteristics of each data sample. dim=(nb_sample x 7)
            7 factors are: "filename", "wave_shape", "amplitude", "freq_range", "is_noise", "scale", "key"
    """
    assert nb_sample > 0
    assert 0.0 <= major_prop <= 1.0

    metadata_df = pd.DataFrame()

    # assign keys for each sample
    nb_major = int(nb_sample * major_prop)
    nb_minor = nb_sample - nb_major
    print(nb_sample, nb_major, nb_minor)
    major_scales = random.choices(sorted(MAJOR_FREQ), k=nb_major)
    minor_scales = random.choices(sorted(MINOR_FREQ), k=nb_minor)
    scales = ["major"]*nb_major + ["minor"]*nb_minor
    keys = major_scales + minor_scales # list

    # assign wave_shape

    if bias != "wave_shape":
        wave_shape = controlling_factors["wave_shape"]
        wave_shapes = [wave_shape] * nb_sample
    
     # assign amplitude
    if bias != "amplitude":
        amplitude = controlling_factors["amplitude"]
        amplitudes = [amplitude] * nb_sample
    
    # assign freq_range
    if bias != "freq_range":
        freq_range = controlling_factors["freq_range"]
        freq_ranges = [freq_range] * nb_sample

    # assign bias
    biases = [bias_type[s] for s in scales] # factor of correlation. CHANGE if needed
    nb_biased_sample = int(nb_sample * bias_strength)
    nb_unbiased_sample = nb_sample - nb_biased_sample
    unbiased_index = random.choices(range(nb_sample), k=nb_unbiased_sample)
    new_biases = [controlling_factors[bias] if i in unbiased_index else biases[i] for i in range(nb_sample)]

    if bias=="wave_shape":
        wave_shapes = new_biases
    elif bias=="amplitude":
        amplitude = new_biases
    else:
        freq_range = new_biases

    # assign noise
    nb_noise_sample = int(nb_sample * noise_level)
    nb_clean_sample = nb_sample - nb_noise_sample
    noise_index = random.choices(range(nb_sample), k=nb_noise_sample)
    is_noises = [True if i in noise_index else False for i in range(nb_sample)]

    # assign filename
    filenames = [data_use + str(f"{i+1:05d}") for i in range(nb_sample)]

    d = {"filename":filenames, "wave_shape":wave_shapes, "amplitude":amplitudes, "freq_range":freq_ranges, "is_noise":is_noises, 
            "scale":scales, "key":keys}
    metadata_df = pd.DataFrame(d)
    print(metadata_df)
    pass

# generate sample according to metadata_file
def data_generation():
    """
    Args: 
        - metadata_csv (csv): csv file containing characteristics of samples to generate
        - noise_level (float): percentage of noise samples in all samples. Range: [0.0, 1.0] 
    
    Return:
        - Audio samples built according to metadata_csv and noise_level
    """
    pass

if __name__ == "__main__":
    FREQ_RANGE = (130.81, 523.25)
    generate_metadata_file(nb_sample=20, major_prop=0.3, bias="wave_shape", 
        bias_type = {"major":"sine", "minor":"square"}, bias_strength=1.0, noise_level = 0.5,
        controlling_factors={"amplitude":"stable", "freq_range": FREQ_RANGE}, data_use="train")

