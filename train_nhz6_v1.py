import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os

import numpy as np
import time
from datasets.dummy_dataset import DummyDataset
from models.NHz6 import *
import torch.optim as optim 

from torch.cuda.amp import autocast, GradScaler
import shutil
import yaml
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        videos = data.get('video').to(device)
        input_len = data.get('input_len').to(device)
        labels = data.get('label').to(device) # Labels should be [batch, seq_len]
        label_lengths = data.get('output_len').to(device) # size: [batch]

        batch_size = videos.size()[0]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(videos,input_len)
        outputs_reshaped = torch.permute(outputs,(1,0,2)) # seq_len, batch, num_classes
                
        # Compute the loss and its gradients
        loss = loss_fn(outputs_reshaped.log_softmax(2).detach().requires_grad_(), labels, input_len, label_lengths)
        loss.requires_grad_().backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_index * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        running_loss = 0.

    return last_loss

def get_data_loaders(args):
    #TODO: si può fare con un solo argomento
    train_ds = DummyDataset(args['video_dir'], args['label_dir'],"train", args)
    val_ds = DummyDataset(args['video_dir'], args['label_dir'],"val", args)
    training_loader = DataLoader(
                train_ds, 
                batch_size=args['batch'], 
                shuffle=True
            )
    validation_loader = DataLoader(
                val_ds, 
                batch_size=args['batch'], 
                shuffle=False
             )
    return training_loader, validation_loader

with open('config.yaml', 'r') as file:
    args = yaml.safe_load(file)

#os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('tensorboard/eLipSys_trainer_{}'.format(timestamp))
model = NHz6(args).to(device)

lr = args['batch'] / 32.0 / torch.cuda.device_count() * args['lr']
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)  
loss_fn = nn.CTCLoss(blank=0, zero_infinity=True) # 0 is conventionally used for blank class in CTC
       


training_loader, validation_loader = get_data_loaders(args)

epoch_number = 0

EPOCHS = args['max_epoch'] 

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            videos = vdata.get('video').to(device)
            input_len = vdata.get('input_len').to(device)
            labels = vdata.get('label').to(device) # Labels should be [batch, seq_len]
            label_lengths = vdata.get('output_len').to(device) # size: [batch]
            
            #vinputs, vlabels = vdata
            voutputs = model(videos,input_len)
            voutputs_reshaped = torch.permute(voutputs,(1,0,2)) # seq_len, batch, num_classes
            vloss = loss_fn(voutputs_reshaped.log_softmax(2).detach().requires_grad_(), labels, input_len, label_lengths)
            running_vloss += vloss



    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'elipsys_{}_{}'.format(timestamp, epoch_number)
        model_path = os.path.join(args['save_prefix'], model_path)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

