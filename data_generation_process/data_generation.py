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

SR = 44100 # sample rate

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
            wavfile.write(f"../../audio/majorr_24/{fname}.wav", SR, wav)
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

def write_metadata(file_name, label, df):
    new_row = pd.DataFrame({'filename': file_name, "label": label})
    try: 
        df.append(new_row, ignore_index=True)
    except NameError:
        df = new_row

    saved_path = "/rds/general/user/cl222/home/audio/metadata" ## CHANGE ME
    df.to_csv(saved_path, sep='\t', index=False)
    return None

######################################################################
def data_generation(scale_type="major", serial_nb = "0001", metadata_df=None):
    FREQ_RANGE = (130.81, 523.25) #c2 to c5

    # sample one frequency
    if scale_type=="major":
        # major scale name
        FREQ_LIST = {"C_":261.6256, 
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
        scale_name = random.sample(list(FREQ_LIST), k=1)[0]
        freq = FREQ_LIST[scale_name]
    else:
        # minor scale name
        FREQ_LIST = {"c_":261.6256, 
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
        scale_name = random.sample(list(FREQ_LIST), k=1)[0]
        freq = FREQ_LIST[scale_name]

    # print("sampled freq: ", freq, '  scale_name: ', scale_name)

    # get notes for scale
    first_note = freq
    second_note = freq*pow(2, 2/12)
    fourth_note = freq*pow(2, 5/12)
    fifth_note = freq*pow(2, 7/12)
    seventh_note = freq*pow(2, 11/12)
    if scale_type=="major":
        third_note = freq*pow(2, 4/12)
        sixth_note = freq*pow(2, 9/12)
    else: # minor notes
        third_note = freq*pow(2, 3/12)
        sixth_note = freq*pow(2, 8/12)
    rest = 0
    scale = np.array([first_note, second_note, third_note, fourth_note, fifth_note, sixth_note, seventh_note, rest])

   # sample number of notes in a melody
    nb_of_notes = random.randrange(7, 13)

    sampled_index = random.choices(range(len(scale)), k=nb_of_notes) # sample with replacement
    sampled_notes = (list(scale[sampled_index]))

    # make sure first note appears in melody
    i = random.randint(0, len(sampled_notes)-1)
    sampled_notes[i] = first_note
    sampled_time = list(np.random.uniform(0.2, 0.55, nb_of_notes))

    sampled_notes2 = copy.deepcopy(sampled_notes)
    sampled_time2 = copy.deepcopy(sampled_time)

    sampled_notes3 = copy.deepcopy(sampled_notes)
    sampled_time3 = copy.deepcopy(sampled_time)
    
    sampled_notes4 = copy.deepcopy(sampled_notes)
    sampled_time4 = copy.deepcopy(sampled_time)

    # add more randomness in bass and root nodes
    i = random.randint(0, 2)
    bass_begin = None
    if i==0:
        bass_begin = first_note
        root_begin = third_note
    else:
        bass_begin = third_note
        root_begin = first_note
    bass_notes = [bass_begin]*len(sampled_notes2)
    root_notes = [root_begin]*len(sampled_notes3)
    seventh_notes = [0.0]*len(sampled_notes4) # additional seventh note

    for i in range(len(sampled_index)):
        # construct minor second or major five
        if (sampled_index[i] == 1 or sampled_index[i] == 5):
            coin = np.random.randint(0,3)
            bass_notes[i] = (sixth_note, fourth_note, seventh_note)[coin]
            root_notes[i] = (second_note, second_note, fifth_note)[coin]
        # construct minor six, major one (weight twice), major forth
        if (sampled_index[i]==0):
            coin = np.random.randint(0,4)
            bass_notes[i] = (third_note, fifth_note, fifth_note, sixth_note)[coin]
            root_notes[i] = (sixth_note, third_note, third_note, fourth_note)[coin]
            seventh_notes[i] = (fifth_note, seventh_note, rest, third_note)[coin]
            coinb = np.random.randint(0,2)
            seventh_notes[i] = (rest, seventh_notes[i])[coinb]
        # diminish seventh or minor third
        if (sampled_index[i]==6):
            coin = np.random.randint(0,2)
            bass_notes[i] = (second_note, fifth_note)[coin]
            root_notes[i] = (fourth_note, third_note)[coin]
    
    ## add noise 
    # to be added

    ## add octive
    for i in range(len(sampled_notes)):
        if sampled_notes[i] !=0.0:
            octives = np.array(get_octive(sampled_notes[i], FREQ_RANGE))
            a = random.sample(range(len(octives)), 1)
            sampled_notes[i] = octives[a]
    for i in range(len(sampled_notes2)):
        octives = np.array(get_octive(bass_notes[i], FREQ_RANGE))
        a = random.sample(range(len(octives)), 1)
        bass_notes[i] = octives[a]
    for i in range(len(sampled_notes3)):
        octives = np.array(get_octive(root_notes[i], FREQ_RANGE))
        a = random.sample(range(len(octives)), 1)
        root_notes[i] = octives[a]
    for i in range(len(sampled_notes4)):
        if seventh_notes[i] !=0.0:
            octives = np.array(get_octive(seventh_notes[i], FREQ_RANGE))
            a = random.sample(range(len(octives)), 1)
            seventh_notes[i] = octives[a]


    gen = WaveAdder(
        ModulatedOscillator(
            SineOscillator(),
            ADSREnvelope(0.01, 0.01, 1, 0.01),
            FrequencyModulator(notes=sampled_notes, note_lens=sampled_time, duration=4.0),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        ModulatedOscillator(
            SineOscillator(),
            ADSREnvelope(0.01, 0.01, 1, 0.01),
            FrequencyModulator(notes=bass_notes, note_lens=sampled_time2, duration=4.0),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        ModulatedOscillator(
            SineOscillator(),
            ADSREnvelope(0.01, 0.01, 1, 0.01),
            FrequencyModulator(notes=root_notes, note_lens=sampled_time3, duration=4.0),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        ModulatedOscillator(
            SineOscillator(),
            ADSREnvelope(0.01, 0.01, 1, 0.01),
            FrequencyModulator(notes=seventh_notes, note_lens=sampled_time4, duration=4.0),
            amp_mod=amp_mod,
            freq_mod = freq_mod_2
        ),
        stereo=False
    )

    wav = gettrig(gen, 0.01+0.01-0.01+4)
    if scale_type == "major":
        file_name = "Test__" + serial_nb ## CHANGE ME
    elif scale_type == "minor":
        file_name = "test__" + serial_nb ## CHANGE ME
    wave_to_file(wav, fname=file_name, scale_type=scale_type)
    new_row = pd.DataFrame([{'filename':file_name, 'label':scale_name}])
    metadata_df = pd.concat([metadata_df, new_row], ignore_index=True)
    return metadata_df

if __name__ == "__main__":
    nb_sample = 1500
    metadata_df = pd.DataFrame()
    for i in range(1, nb_sample+1):
        metadata_df = data_generation(scale_type="major", serial_nb=f"{i:04d}", metadata_df=metadata_df)
        metadata_df = data_generation(scale_type="minor", serial_nb=f"{i:04d}", metadata_df=metadata_df)
    saved_path = "/rds/general/user/cl222/home/audio/metadata_24.csv" ## CHANGE ME
    print(metadata_df)
    metadata_df.to_csv(saved_path, sep='\t', index=False)