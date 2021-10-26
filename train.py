import os
import torch

import pandas as pd
import torchaudio
import wandb
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torchsummary import summary

from process_data import clear_file, get_data
from resnet50_model import resnet
from process_data import UrbanSoundDataset
from predict import predictions
'''
# Log in to your W&B account
api_key=
wandb.login(key=[api_key])

wandb.init(project='resnet50', config={
          # "layer_1": 512,
          # "activation_1": "relu",
          # "dropout": random.uniform(0.01, 0.80),
          # "layer_2": 10,
          # "activation_2": "softmax",
          "k_folds" : 5,
          "optimizer": "Adam",
          "loss": "nn.CrossEntropyLoss()",
          "metric": "accuracy",
          "epoch": 10,
          "batch_size": 128,
          "learning_rate" : 0.001
      })

config = wandb.config
'''
'''
from ignite.handlers import ModelCheckpoint
from ignite.handlers import EarlyStopping
from ignite.engine import Engine, Events

# Checkpoint
checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                      save_top_k=1,
                                      save_last=True,
                                      save_weights_only=True,
                                      filename_prefix='/content/drive/MyDrive/Cogito HE Challenge/checkpoint/{epoch:02d}-{valid_loss:.4f}-{valid_f1:.4f}',
                                      verbose=False,
                                      mode='min',
                                      dirname='/content/drive/MyDrive/Cogito HE Challenge')

def score_function(engine):
    val_loss = engine.state.metrics['nll']
    return -val_loss


# Earlystopping
earlystopping = EarlyStopping(patience=3, score_function=score_function, trainer='valid_acc')
'''


BATCH_SIZE = 128
EPOCHS = 1
learning_rate = 0.001
k_folds=5

DIR = "data/"
ANNOTATIONS_FILE = "data/train.csv"
AUDIO_DIR = "data/train/"
new_annpath = DIR+'cleaned_annotation.csv'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

TEST_ANNOTATIONS_FILE = 'data/test500.csv'

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, test_loader, loss_fn, optimiser, device):
    correct = 0
    total = 0
    for input, target in data_loader:
        input = input.to(device)
        target = target.to(device)
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        _, predicted = torch.max(prediction.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"loss: {loss.item()}")
    accuracy = 100 * correct / total         #4810               #len(trainset)
    print("Train Accuracy = {}".format(accuracy))

    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)
            # calculate loss
            prediction = model(input)
            _, predicted = torch.max(prediction.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Val Accuracy: ', 100.0 * correct / total)
    result = 100.0 * (correct / total)

    scheduler.step()

    # update progress bar
    #pbar.update(pbar_update)

    return loss, accuracy, result


def train(model, data_loader, test_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        loss, accuracy, result = train_single_epoch(model, data_loader, test_loader, loss_fn, optimiser, device)
        #wandb.log({'epoch': epochs+1, 'loss': loss, 'accuracy': accuracy, 'val_acc': result})
        print("---------------------------")
    print("Finished training")
    return result


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using {device}")

    new_ann = clear_file(ANNOTATIONS_FILE, DIR, new_annpath)

    filenames, labels = get_data(new_ann, AUDIO_DIR)

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    print('')
    
    #print('USD: ',len(usd))

    # construct model and assign it to device
    cnn = resnet(device)
    cnn = cnn.to(device)
    print(cnn) 
    #wandb.watch(cnn, log="all")
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=learning_rate, weight_decay=0.0001
                                 )
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
    
    #train_dataloader = create_data_loader(usd, BATCH_SIZE)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # train model
    #train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)
    # For fold results
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(filenames)):
        train_first, train_last = train_ids[0], train_ids[-1]
        test_first, test_last = test_ids[0], test_ids[-1]
        train_filenames = filenames[train_first: train_last]
        train_labels = labels[train_first: train_last]
        test_filenames = filenames[test_first: test_last]
        test_labels = labels[test_first: test_last]
        
        train_usd = UrbanSoundDataset(train_filenames, train_labels, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
        train_dataloader = create_data_loader(train_usd, BATCH_SIZE)
        test_usd = UrbanSoundDataset(test_filenames, test_labels, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
        test_dataloader = create_data_loader(test_usd, BATCH_SIZE)
        pbar_update = 1 / (len(train_dataloader) + len(test_dataloader))
        print('Training Fold: ',fold)
        results[fold] = train(cnn, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")

    class_mapping = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    exp, prr = predictions(TEST_ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, class_mapping, device)

'''
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=exp, preds=prr,
                            class_names=class_mapping)})


    wandb.log({"displot" : sns.distplot(exp, prr)})

    wandb.log({"scatterplot" : plt.scatter(exp, prr)})                                   #It shoud be linear

    wandb.finish()
'''









