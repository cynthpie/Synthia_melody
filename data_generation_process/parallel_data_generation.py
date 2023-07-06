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
from joblib import Parallel, delayed

print("data generation script used to produce neutral 10000 test data 001")
# define constants
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

# amplitude class (for now)
STABLE = (0.01, 0.01, 1, 0.01) # attack_duration, decay_duration, sustain_level, release_duration
INCREASE = (2, 0.01, 1, 0.01)
DECREASE = (0.01, 0.01, 1, 2)


#####################################################################
# Helper functions
hz = lambda note:librosa.note_to_hz(note)
to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))

def wave_to_file(wav, wav2=None, fname="temp", amp=0.1):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    try:
        wavfile.write(f"../../audio/square_test/{fname}.wav", SR, wav)
    except FileNotFoundError:
        os.makedirs('../../audio/square_test/')
        wavfile.write(f"../../audio/square_test/{fname}.wav", SR, wav)

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
    sampled_oct = random.sample(octive_list, k=1)[0]
    return sampled_oct

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
    major_scales = random.choices(sorted(MAJOR_FREQ), k=nb_major)
    minor_scales = random.choices(sorted(MINOR_FREQ), k=nb_minor)
    scales = ["major"]*nb_major + ["minor"]*nb_minor
    keys = major_scales + minor_scales # list

    # assign wave_shape
    wave_shape = controlling_factors["wave_shape"]
    wave_shapes = [wave_shape] * nb_sample
    # assign amplitude
    amplitude = controlling_factors["amplitude"]
    amplitudes = [amplitude] * nb_sample
    # assign freq_range
    freq_range = controlling_factors["freq_range"]
    freq_ranges = [freq_range] * nb_sample

    # assign bias
    if bias_strength >0:
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
    return metadata_df

# generate sample according to metadata_file
def data_generation(one_metadata):
    """
    Args: 
        - one_metadata (pandas.core.series.Series): one sample in metadata_df
    Return:
        - An audio sample built according to one_metadata 
    """
    # load info
    filename = one_metadata["filename"]
    scale = one_metadata["scale"]
    key = one_metadata["key"]
    wave_shape = one_metadata["wave_shape"]
    amplitude = one_metadata["amplitude"]
    freq_range = one_metadata["freq_range"]
    is_noise = one_metadata["is_noise"]

    # build scale and triads
    freq = MAJOR_FREQ[key] if scale=="major" else MINOR_FREQ[key]  
    first_note = freq  
    second_note = freq*pow(2, 2/12)
    fourth_note = freq*pow(2, 5/12)
    fifth_note = freq*pow(2, 7/12)
    seventh_note = freq*pow(2, 11/12)
    if scale=="major":
        third_note = freq*pow(2, 4/12)
        sixth_note = freq*pow(2, 9/12)
    else: # minor notes
        third_note = freq*pow(2, 3/12)
        sixth_note = freq*pow(2, 8/12)
    rest = 0
    scale = np.array([first_note, second_note, third_note, fourth_note, fifth_note, sixth_note, seventh_note, rest])
    triads = np.array(["first_triad", "second_triad", "third_triad", "fourth_triad", "fifth_triad", "sixth_triad", "seventh_triad"])

    # sample number of notes in a melody
    nb_of_traids = random.randrange(3, 7)
    sampled_index = random.choices(range(len(triads)), k=nb_of_traids) # sample with replacement
    sampled_triads = triads[sampled_index]

    # control noise level
    if not is_noise:
        available_index = list(range(len(sampled_triads)))
        while ("first_triad" not in sampled_triads) or ("fourth_triad" not in sampled_triads) or ("fifth_triad" not in sampled_triads):
            if ("first_triad" not in sampled_triads):
                i = random.sample(available_index, k=1)
                i = np.array(i)
                sampled_triads[i] = "first_triad"
                available_index.remove(i)
            if ("fourth_triad" not in sampled_triads):
                i = random.sample(available_index, k=1)
                i = np.array(i)
                sampled_triads[i] = "fourth_triad"
                available_index.remove(i)
            if ("fifth_triad" not in sampled_triads):
                i = random.sample(available_index, k=1)
                i = np.array(i)
                sampled_triads[i] = "fifth_triad"
                available_index.remove(i)
    else:
        available_index = list(range(len(sampled_triads)))
        available_triad = ["second_triad", "third_triad", "sixth_triad", "seventh_triad"]
        while ("first_triad" in sampled_triads) and ("fourth_triad" in sampled_triads) and ("fifth_triad" in sampled_triads):
            i = random.randint(0, len(available_index)-1)
            j = random.randint(0, len(available_triad)-1)
            sampled_triads[i] = available_triad[j]
            available_index.remove(i)

    # build notes with sampled chords
    ch_1_notes = [first_note if traid in ("first_triad", "fourth_triad", "sixth_triad") else seventh_note if traid in ("third_triad")\
        else second_note for traid in sampled_triads]
    ch_2_notes = [third_note if traid in ("first_triad", "third_triad", "sixth_triad") else seventh_note if traid in ("fifth_triad")\
        else fourth_note for traid in sampled_triads]
    ch_3_notes = [fifth_note if traid in ("first_triad", "third_triad", "fifth_triad") else seventh_note if traid in ("seventh_triad")\
        else sixth_note for traid in sampled_triads]
    coin = random.randint(0,1)
    if coin==1:
        ch_4_notes = [first_note if traid in ("second_triad") else fourth_note if traid in ("fifth_triad") \
            else sixth_note if traid in ("seventh_triad") else rest for traid in sampled_triads]   # seventh channel 2_7, 5_7, 7_7
    else: ch_4_notes = [0.0] *len(ch_3_notes)

    # sample duration of notes
    sampled_time = list(np.random.uniform(0.2, 0.9, nb_of_traids))
    sampled_time2, sampled_time3, sampled_time4 = copy.deepcopy(sampled_time), copy.deepcopy(sampled_time), copy.deepcopy(sampled_time)

    # add octives 
    ch_1_notes = [get_octive(note, freq_range) for note in ch_1_notes]
    ch_2_notes = [get_octive(note, freq_range) for note in ch_2_notes]
    ch_3_notes = [get_octive(note, freq_range) for note in ch_3_notes]
    ch_4_notes = [get_octive(note, freq_range) if note > 0.0 else 0.0 for note in ch_4_notes]


    # build melody
    oscillator = SineOscillator if wave_shape=="sine" else SquareOscillator if wave_shape=="square" \
        else SawtoothOscillator if wave_shape=="sawtooth" else TriangleOscillator
    amp_change = STABLE if amplitude=="stable" else INCREASE if amplitude=="increase" else DECREASE
    gen = WaveAdder(
        ModulatedOscillator(
            oscillator(sample_rate=SR),
            ADSREnvelope(amp_change[0], amp_change[1], amp_change[2], amp_change[3], sample_rate=SR),
            FrequencyModulator(notes=ch_1_notes, note_lens=sampled_time, duration=4.0, sample_rate=SR),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        ModulatedOscillator(
            oscillator(sample_rate=SR),
            ADSREnvelope(amp_change[0], amp_change[1], amp_change[2], amp_change[3], sample_rate=SR),
            FrequencyModulator(notes=ch_2_notes, note_lens=sampled_time2, duration=4.0, sample_rate=SR),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        ModulatedOscillator(
            oscillator(sample_rate=SR),
            ADSREnvelope(amp_change[0], amp_change[1], amp_change[2], amp_change[3], sample_rate=SR),
            FrequencyModulator(notes=ch_3_notes, note_lens=sampled_time3, duration=4.0, sample_rate=SR),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        ModulatedOscillator(
            oscillator(sample_rate=SR),
            ADSREnvelope(amp_change[0], amp_change[1], amp_change[2], amp_change[3], sample_rate=SR),
            FrequencyModulator(notes=ch_4_notes, note_lens=sampled_time4, duration=4.0, sample_rate=SR),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        stereo=False
    )
    
    # store to wav file
    wav = gettrig(gen, amp_change[0]+amp_change[1]-amp_change[3]+4) # check
    wave_to_file(wav, fname=filename)
    return None

if __name__ == "__main__":
    FREQ_RANGE = (130.81, 523.25)
    NB_SAMPLE = 10000
    metadata_df = generate_metadata_file(nb_sample=NB_SAMPLE, major_prop=0.5, bias="wave_shape", 
            bias_type = {"major":"square", "minor":"sine"}, bias_strength=0.0, noise_level = 0.0,
            controlling_factors={"amplitude":"stable", "freq_range": FREQ_RANGE, "wave_shape":"square"},
            data_use="test")
    saved_path = "/rds/general/user/cl222/home/audio/metadata_square_test.csv" ## CHANGE ME!!!!!!!!!!!
    metadata_df.to_csv(saved_path)
    print(metadata_df)
    Parallel(n_jobs=6)(delayed(data_generation)(metadata_df.iloc(0)[i]) for i in range(NB_SAMPLE))
    # data_generation(metadata_df.iloc(0)[1])

