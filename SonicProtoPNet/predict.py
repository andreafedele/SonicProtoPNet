import os
import torch
import argparse
import torchaudio
from sklearn.metrics import classification_report

from audio_dataset import AudioDataset
from torch.autograd import Variable

# parsing model directory and model name to load
parser = argparse.ArgumentParser()
parser.add_argument('-model_dir', nargs=1, type=str, default='0')
parser.add_argument('-model_name', nargs=1, type=str, default='0')
args = parser.parse_args()

model_dir = args.model_dir[0]
model_name =  args.model_name[0] 
model_path = os.path.join(model_dir, model_name)
# model_path = './saved_models/vgg19/007/10_0push0.9571.pth'

# loading the model
ppnet = torch.load(model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# loading the test set
from settings import test_dir, test_annotation_dir, sample_rate, num_samples, n_fft, hop_length, n_mels, power_or_db, test_batch_size

mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

cut_dimensions = (n_mels, n_mels)
test_dataset = AudioDataset(test_annotation_dir, test_dir, sample_rate, num_samples, mel_spectrogram_transformation, power_or_db, cut_dimensions)
print(f"There are {len(test_dataset)} samples in the test dataset.")

y_true, y_pred = [], []
for x, y in test_dataset:
    x_tensor = torch.tensor(x)
    x = Variable(x_tensor.unsqueeze(0))
    x = x.cuda()

    y_torch = torch.tensor([int(y)])

    logits, min_distances = ppnet_multi(x)
    tables = []
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), y_torch[i].item()))

    predicted_cls = tables[0][0]
    y_pred.append(predicted_cls + 1) # dovuto al fatto che lui le conta a partire da 0 nella pre-process function di train_and_test.py
    y_true.append(int(y))

cr = classification_report(y_true, y_pred)
print(cr)

# python predict.py -model_dir './saved_models/vgg19/001_esc50_split10/' -model_name '17push0.7400.pth'