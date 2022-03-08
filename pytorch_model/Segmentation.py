# Train Patch Convolutional AutoEncoder
# Felipe Giuste
# 09-04-2020
# Modified by Danni Chen 02/22/2022

import numpy as np
import pandas as pd
from datetime import datetime

## PyTorch ##
from torch import nn, optim

## Tensorboard ##
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

## Utils ##
from utils.EarlyStopping import EarlyStopping
from torch.utils.data import DataLoader

### User Variables ###
# ----------------------------------------
## Model Name ##
timeStamp = str(datetime.now())
model_name = 'UNET_SEGMENTATION_COVID_LUNG ' + timeStamp

## Import Model, Distance ##
from utils.TrainTestSplit import ImageDataset, train_test_split
from utils.UNET_model import UNet
from UNET_Train_COPY import UNET_Train
from utils.dice_score import *

metadata = pd.read_csv("metadata.csv")
slice_cols = "CT_image_path"
n_channels = 1
n_classes = 2
batch_size = 8
numOfEpoch = 1
# -----------------------------------------

### Setup Device ###
## Use best Device (CUDA vs CPU) ##
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # Print Device Properties ##
# if device == torch.device('cuda'):
#     print(torch.cuda.get_device_properties(device))

device = "cpu"

## Command Line Argument Processing ##
import sys

debugger = False
if len(sys.argv) > 1:
    print(sys.argv)
    # Use Debugger
    try:
        debugger = sys.argv[sys.argv.index('--debug')]
        print('(User Defined) Use Debugger: %s' % debugger)
    except:
        pass
## Use Debugger ##
if debugger:
    import pdb

    pdb.set_trace()

### Seed ###
random_state = 1234
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
## CUDNN ##
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

## Weight of Reconstruction Loss ##
alpha_reconstruction = 1.

# ## Early Stopping Patience ##
patience = 100
## Total Epochs ##
# n_epochs = 300
# n_epochs = 200 # V16-17
## Images per batch ##
batch_size = 8  # V10-

### Data Load ###
print('')
# Retrieve the dataset from info obtained in metadata dataframe
dataset = ImageDataset(metadata=metadata, root='.', transform=None, slice_cols=slice_cols)

# Split a ImageDataset into two groups (training and testing) based on information specified within its metadata
# train_test_split returns a tuple containing two ImageDataset objects with training and testing data, respectively.
(training_data, testing_data) = train_test_split(dataset)

print('Number of data in the training dataset: ' + str(len(training_data)))
print('Number of data in the testing dataset: ' + str(len(testing_data)) + '\n')

# Load the data with dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

### Tensorboard ###
## Save data for Tensorboard ##
writer = SummaryWriter('runs/UNET/%s' % model_name)
## Start Tensorboard on port 6006 ##
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './runs/UNET/', '--host', '0.0.0.0', '--reload_interval', '60'])
url = tb.launch()

### Model Setup ###
# Model definition and loading
model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)

# Use the generic prepared class to handle model training/testing and data capture
lungMaskUnet = UNET_Train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9),
    writer=writer,
    device=device,
    verbose=True
)

## Model: Use multiple GPUs ##
if torch.cuda.device_count() > 1:
    print("Total GPUs: %s" % torch.cuda.device_count())
    model = nn.DataParallel(model)
## Model: Send to device ##
model.to(device)
## Print model ##
# print(model)

### Model Train ###
lungMaskUnet.model_training(numOfEpoch=numOfEpoch, model_name=model_name)
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True, path='model/%s_checkpoint.pt' % model_name)

## Empty GPU Cache ##
torch.cuda.empty_cache()
## End Tensorboard Writer ##
writer.close()

### Save Model State: Final ###
torch.save(model.state_dict(), 'model/%s_Final.pt' % model_name)

# Release resources.
lungMaskUnet.close()
