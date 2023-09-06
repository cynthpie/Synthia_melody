# Synthia's melody
An audio-data generation mechanism for distribution shifts. 

### Data generation
To generate data, run "parallel_data_generation.py" in the folder "data_generation". Change the argparse setting to the desired configuration. 
- Ex. run ```python parallel_data_generation.py --nb_sample="5" --waveshape="square" --amplitude="stable"```.
- There will be two output files:
  1. The metadata.csv. The csv file contains filenames of the generated audio and their music key label in 24 major/minor keys. Major keys are denoted with uppercase and minor keys are in lowercase letters.
  2. A folder containing the generated audio file. The audios are in the form of ```.wav``` file and have a sample rate of 16000 Hz.

The default timbre

