#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Utils
from utils.TrainTestSplit import ImageDataset, train_test_split
from utils.UNET_model import UNet
from UNET_Train import UNET_Train
from utils.dice_score import *

# User’s variables
# ----------------------------------------
## Model Name ##
timeStamp = str(datetime.now())
model_name = 'UNET_SEGMENTATION_COVID_LUNG ' + timeStamp
metadata = pd.read_csv("metadata.csv")
slice_cols = "CT_image_path"
# Select a device.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_channels = 1
n_classes = 2
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
batch_size = 1
numOfEpoch = 10
# -----------------------------------------

#  Retrieve the dataset from info obtained in metadata dataframe
dataset = ImageDataset(metadata=metadata, root='.', transform=None, slice_cols=slice_cols)

# Split a ImageDataset into two groups (training and testing) based on information specified within its metadata
# dataframe Return a tuple containing two ImageDataset objects with training and testing data, respectively.
(training_data, testing_data) = train_test_split(dataset)

print('Number of data in the training dataset: ' + str(len(training_data)))
print('Number of data in the testing dataset: ' + str(len(testing_data)) + '\n')

# Load the data with dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

# Model definition and loading
model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)

## Setup Device ###
# Use best Device (CUDA vs CPU) ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Print Device Properties ##
if device == torch.device('cuda'):
    print(torch.cuda.get_device_properties(device))

model = model.to(device)

# Use the generic prepared class to handle model training/testing and data capture
lungMaskUnet = UNET_Train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    # criterion=dice_coeff(),
    optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9),
    # writer=SummaryWriter(log_dir=log_dir),
    writer=None,
    device=device,
    verbose=True
)

# Train the model and capture the statistics of the model at each epoch.
stats = lungMaskUnet.model_training(numOfEpoch=numOfEpoch)

## Generate and save the train_test loss plot
training_loss = stats['train_loss']
validation_loss = stats['test_loss']

plt.plot(range(len(training_loss)), training_loss, color='green', label='train loss')
plt.plot(range(len(validation_loss)), validation_loss, color='red', label='val loss')
plt.legend()
plt.title("UNET Model (LUNG) train & val loss plot")
plt.show()

plt.savefig('Train_Val_LOSS-%s.png ' % timeStamp)
