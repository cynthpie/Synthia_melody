# Synthia's melody
An audio data generation mechanism for distribution shifts. 

Example usage:
<img width="1157" alt="synthia_chart" src="https://github.com/cynthpie/msc_project/assets/134090009/f946b0cc-36a2-4596-bd22-b9d73261b2c3">



### Data generation
To generate data, run "parallel_data_generation.py" in the folder "data_generation". Change the argparse setting to the desired configuration. 
- Ex. run ```python parallel_data_generation.py --nb_sample="5" --waveshape="square" --amplitude="stable"```.
- There will be two output files:
  1. The metadata.csv. The csv file contains filenames of the generated audio and their sampled music key label in 24 major/minor keys. Major keys are denoted with uppercase and minor keys are in lowercase letters.
  2. A folder containing the generated audio file. The audios are in the form of ```.wav``` file and have a sample rate of 16000 Hz.


### Parameters
The parameters of the data generation mechanism are:
- nb_sample: number of sample to generate.
- waveshape: can be "sine", "square", "sawtooth", "triangle". Different waveshape results in different music timbre.
- amplitude: can be "stable", "increase", or "decrease". The loudness of melody will vary according to the description.
- freq_lower: the lowest frequency (pitch) to generate in Hz.
- freq_upper: the highest frequency (pitch) to generate in Hz.
- major_prop: proportion of major keys in the generation data. 
- noise_level: proportion of samples exposed to label noise. e.g. samples without all 7 notes of its key shown in melody.
- seed: random seed to generate melody samples 

The code in the folder "synth" is taken from Tom, Alan (2021) availiable at https://github.com/18alantom/synth.
