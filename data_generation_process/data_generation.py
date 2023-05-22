# data generation process
import numpy as np
import copy
import random
from scipy.io import wavfile
import librosa
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

def wave_to_file(wav, wav2=None, fname="temp", amp=0.1, tune="major"):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    if tune == "major":
        wavfile.write(f"tempsnd/major/{fname}.wav", SR, wav)
    else: 
        wavfile.write(f"tempsnd/minor/{fname}.wav", SR, wav)

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


######################################################################
def data_generation():
    FREQ_RANGE = (130.81, 523.25) #c2 to c5
    NB_SAMPLE = 10
    CHORD = "major" # major or minor

    # sample one frequency
    freq = np.random.uniform(FREQ_RANGE[0], FREQ_RANGE[1])

    # get notes for chord
    first_note = freq
    second_note = freq*pow(2, 2/12)
    third_note = freq*pow(2, 4/12)
    fourth_note = freq*pow(2, 5/12)
    fifth_note = freq*pow(2, 7/12)
    sixth_note = freq*pow(2, 9/12)
    seventh_note = freq*pow(2, 11/12)
    rest = 0

    scale = np.array([first_note, second_note, third_note, fourth_note, fifth_note, sixth_note, seventh_note])
    sampled_index = random.sample(range(len(scale)),7)
    sampled_notes = (list(scale[sampled_index]))
    i = random.randint(0, (len(sampled_notes)-1))
    sampled_notes[i] = third_note
    sampled_notes.append(first_note)
    sampled_time = list(np.random.uniform(0.3, 0.6, 8))

    sampled_notes2 = copy.deepcopy(sampled_notes)
    sampled_time2 = copy.deepcopy(sampled_time)

    sampled_notes3 = copy.deepcopy(sampled_notes)
    sampled_time3 = copy.deepcopy(sampled_time)
    root_notes = [freq/2]*len(sampled_notes3)

    bass_notes = [freq]*len(sampled_notes2)
    for i in range(len(sampled_index)):
        if (sampled_index[i] == 1 or sampled_index[i] == 5) and sampled_time[i] > 0.4:
            coin = np.random.randint(0,2)
            bass_notes[i] = (second_note, fourth_note)[coin]
            root_notes[i] = second_note/2
        if (sampled_index[i]==0) and sampled_time[i] > 0.5:
            coin = np.random.randint(0,2)
            bass_notes[i] = ( third_note, sixth_note)[coin]
            root_notes[i] = first_note/2

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
        stereo=False
    )

    wav = gettrig(gen, 0.01+0.01-0.01+4)
    file_name="Test__"+str(b) ## CHANGE ME
    wave_to_file(wav, fname=file_name, tune="major")
    print(file_name)