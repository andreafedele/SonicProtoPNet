import os
import librosa
import numpy as np

def feature_melspectrogram(waveform, sample_rate):
    return np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128).T,axis=0)

def feature_chromagram(waveform, sample_rate):
    return np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(waveform)), sr=sample_rate).T,axis=0)

def feature_mfcc(waveform, sample_rate):    
    return np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0) 

def feature_spectral_centroid(waveform, sample_rate):
    return librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)

def load(path, split, type='spect'):
    X, y, filenames = [], [], []

    for folder in os.listdir(path + split):
        if folder == '.DS_Store':
            continue
            
        for file in os.listdir(path + split + folder):
            yy, sr = librosa.load(path + split + folder + '/' + file)

            if type == 'cent':
                feature = feature_spectral_centroid(yy, sr)
            elif type == 'spect':
                feature = feature_melspectrogram(yy, sr)
            else:
                chromagram = feature_chromagram(yy, sr)
                melspectrogram = feature_melspectrogram(yy, sr)
                mfc_coefficients = feature_mfcc(yy, sr)
                feature = np.hstack((chromagram, melspectrogram, mfc_coefficients))
                
            X.append(feature)
            y.append(folder)
            filenames.append(path + split + folder + '/' + file)
    
    return np.array(X), np.array(y), np.array(filenames)

train = 'train/'
test = 'test/'

datasets_folder = ['Medley-solos-DB_split/', 'GTZAN_split/', 'esc50_split/']

for dataset_folder in datasets_folder:
    print("DATASET: ", dataset_folder)
    path = '../datasets/' + dataset_folder

    for folder_split in os.listdir(path):
        if folder_split == '.DS_Store':
            continue

        export_path = path + folder_split + '/ts/'
        if not os.path.exists(export_path):
            os.mkdir(export_path)

        for type in ['cent', 'spect', 'all']:
            X_train, y_train, filenames_train = load(path + folder_split + '/', train, type=type)
            X_test, y_test, filenames_test = load(path + folder_split + '/', test, type=type)
            
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, filenames_train.shape, filenames_test.shape)
    
            vals_to_save = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'filenames_train': filenames_train,
                'filenames_test': filenames_test
            }

            print(export_path + type + '.npz')
            np.savez(export_path + type + '.npz', **vals_to_save)