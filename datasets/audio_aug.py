import os
import time
import torch
import random
import librosa
import numpy as np
import pandas as pd

import torchaudio
import torchaudio.functional as F
from torchaudio.utils import download_asset

SPLIT_DIR = 'DATASET_DIR_HERE'

AUG_PER_SAMPLE = 1 # how many augmented samples to generate from each audio file
AUGMENTATION_STRATEGY = 'snp' # 's' time stretch, 'n' background noise, 'pitch' pitch shift 
STRETCH_RATE_RANGE = (0.8, 0.9)
NOISE_DB_RANGE = (0, 10)
SEMITONES_SHIFT_RANGE = (1, 7)

# how many augmented samples to generate for each augmentation strategy
AUG_PER_STRATEGY = AUG_PER_SAMPLE if AUG_PER_SAMPLE else round(AUG_PER_SAMPLE/len(AUGMENTATION_STRATEGY))

SAMPLE_NOISE_PATH = "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav" 
NOISE, _ = torchaudio.load(download_asset(SAMPLE_NOISE_PATH))
NOISE = torch.cat((NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE), dim=1) # triplicating NOISE duration by concatenation

# --------- UTILS FUNCTIONS -----------
def create_directory_if_necessary(path):
    if not os.path.exists(path):
        os.mkdir(path)

def augment_noise(signal):
    noise = NOISE[:, : signal.shape[1]] # reshaping noise len based on signal shape
    dB_bins = np.linspace(NOISE_DB_RANGE[0], NOISE_DB_RANGE[1], AUG_PER_STRATEGY)
    snr_dbs = torch.tensor(dB_bins, dtype=torch.float32)
    noisy_audios = F.add_noise(signal, noise, snr_dbs)

    return [noisy_audios[i:i+1] for i in range(len(noisy_audios))]

def agument_pitch_shift(signal, sr):
    semitones_bins = np.linspace(SEMITONES_SHIFT_RANGE[0], SEMITONES_SHIFT_RANGE[1], AUG_PER_STRATEGY)
    return [torch.tensor(librosa.effects.pitch_shift(signal.numpy()[0], sr=sr, n_steps=semitones_bins[i])[None, :], dtype=torch.float32) for i in range(len(semitones_bins))]

def augument_time_stretch(signal):
    stretch_bins = np.linspace(STRETCH_RATE_RANGE[0], STRETCH_RATE_RANGE[1], AUG_PER_STRATEGY)
    return [torch.tensor(librosa.effects.time_stretch(signal.numpy()[0], rate=stretch_bins[i])[None, :], dtype=torch.float32) for i in range(len(stretch_bins))]
# --------------------------------------

create_directory_if_necessary(SPLIT_DIR + '/train_augmented')

data_augmented = {'file_name': [], 'label': []}
for class_label in os.listdir(SPLIT_DIR + '/train'):
    st = time.time()
    print("Augmenting class label:", class_label)
    if class_label == '.DS_Store':
        continue

    audios = os.listdir(SPLIT_DIR + '/train/' + class_label)
    create_directory_if_necessary(SPLIT_DIR + '/train_augmented/' + class_label)

    for audio in audios:
        # load original audio
        signal, sr = torchaudio.load(SPLIT_DIR + '/train/' + class_label + '/' + audio)
        augmented_audios = []

        # get augmented audios, each function returns an AUG_PER_STRATEGY number of augmented audios 
        if 'n' in AUGMENTATION_STRATEGY:
            augmented_audios += augment_noise(signal)
        if 'p' in AUGMENTATION_STRATEGY:
            augmented_audios += agument_pitch_shift(signal, sr)
        if 's' in AUGMENTATION_STRATEGY:
            augmented_audios += augument_time_stretch(signal)

        # filtering only on 1 random augmentation if 1 AUG_PER_SAMPLE is required
        if AUG_PER_SAMPLE == 1:
            augmented_audios = [random.choice(augmented_audios)]

        for i in range(len(augmented_audios)):
            augmented_audio_file_name = audio[0:-4] + '_' + str(i) + '.wav'
            data_augmented['file_name'].append(augmented_audio_file_name)
            data_augmented['label'].append(class_label)

            torchaudio.save(SPLIT_DIR + '/train_augmented/' + class_label + '/' + augmented_audio_file_name, augmented_audios[i], sr)
    
    print("--- %s seconds ---" % (time.time() - st))


df_augmented = pd.DataFrame(data_augmented)
df_augmented.to_csv(SPLIT_DIR + '/annotations_train_augmented.csv', index=False)
