# SonicProtoPNet

## SonicProtoPNet folder
SonicProtoPNet code is provided. 
Entry point of the whole code is the 'main.py' file, that must be run after proper setting of the 'settings.py' (dataset folders paths, spectrograms conversion params).
Train can be launched after the dataset has been 1) downloaded 2) split and 3) augmented - described in the following paragraphs -.

Local classification analysis launch using script example:
python local_analysis_audio.py -test_model_dir './saved_models/vgg19/001_esc50_split_10/' -test_model_name '10_4push0.9000.pth' -test_spect_dir '../datasets/esc50_split_10/test/9/' -test_spect_name '2-81112-A-34.wav' -test_spect_label 9

where:
- test_model_dir: saved model directory 
- test_model_name: name/weights of the model to use (the user can decide the weights from each push epochs happened during training)
- test_spect_dir: directory of the test audio to classify/analyse/soundify
- test_spect_name: file name of the test audio 
- test_spect_label: label of the test audio 

Log files of our experiments SonicProtoPNet experiments and setting parameters are also provided.


## Dataset
ESC-50 can be downloaded from:
https://github.com/karolpiczak/ESC-50

GTZAN can be downloaded from:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

Medley-solos-DB can be downloaded from:
https://zenodo.org/records/3464194

- /datasets: 
scripts to recreate the class splits dataset used in the paper;
links to the dataset download;
scripts to augment the splitted folders (pitch, noise, stretch).
the order to follow is: 1) download the dataset 2) train/test split 3) augmentation

## kNN
- /kNN: 
script to recreate the dataset representation used for kNN in the paper;
script to run the experiments;
log file of kNN experiments;
kNN dataset generation must be run after dataset download and train test split previously described (steps 1, 2).
