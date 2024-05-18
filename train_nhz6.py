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


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

with open('config.yaml', 'r') as file:
    args = yaml.safe_load(file)

#os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_model = NHz6(args).to(device)
    
if len(args['gpus']) > 1:
    print("parallelling model...")
    video_model =  nn.DataParallel(video_model)

lr = args['batch'] / 32.0 / torch.cuda.device_count() * args['lr']
optimizer = optim.Adam(video_model.parameters(), lr = lr, weight_decay=1e-4)     
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=40, threshold=0.001)


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def train():
    
    max_epoch = args['max_epoch']    
    best_loss = np.nan

    tot_iter = 0
    scaler = GradScaler()     
    tic = time.time()  

    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # 0 is conventionally used for blank class in CTC
    
    for epoch in range(max_epoch):
        for phase in ['train', 'val']:
            dataset = DummyDataset(args['video_dir'], args['label_dir'],[phase], args)
            print(f'Start {phase} phase, dataset size:',len(dataset))
            #loader = dataset2dataloader(dataset, args['batch'], args['num_workers'], mode=phase)
            loader = DataLoader(
                dataset, 
                batch_size=args['batch'], 
                shuffle=True
            )
            for (i_iter, input) in enumerate(loader):
                         
                videos = input.get('video').to(device)
                input_len = input.get('input_len').to(device)
                labels = input.get('label').to(device) # Labels should be [batch, seq_len]
                label_lengths = input.get('output_len').to(device) # size: [batch]
                #with autocast():
                output = video_model(videos,input_len)  # Model outputs are [batch, seq_len, num_classes]                
                output_reshaped = torch.permute(output,(1,0,2)) # seq_len, batch, num_classes
                # Here, the targets are labels, input_lengths are output_lengths, and target_lengths are label_lengths
                loss = criterion(output_reshaped.log_softmax(2).detach().requires_grad_(), labels, input_len, label_lengths)
                 
                if phase == 'train':
                    """ optimizer.zero_grad()   
                    scaler.scale(loss).backward()  
                    scaler.step(optimizer)
                    scaler.update() """

                    # Backward pass and optimize
                    loss.requires_grad_().backward()
                    optimizer.step()

                    toc = time.time()
                    msg = 'epoch={}, train_iter={}, eta={:.1f}s'.format(epoch, tot_iter, (toc-tic)*(len(loader)-i_iter))     
                    tic = time.time()                                          
                    msg += ', train loss={:.5f}'.format(loss)
                    msg = msg + str(', lr=' + str(showLR(optimizer)))                 
                    msg = msg + str(', best valid loss={:2f}'.format(best_loss))
                    print(msg)
                    tot_iter += 1 
                else:
                    valid_loss = loss
                    savename = os.path.join(args['save_prefix'], 'last.pt')
                    if not os.path.exists(args['save_prefix']):
                        os.makedirs(args.save_prefix)
                    print("saving model at " + savename)
                    if len(args.gpus) > 1:
                        torch.save(
                            {
                                'video_model': video_model.module.state_dict(),
                            }, savename)  
                    else:
                        torch.save(
                            {
                                'video_model': video_model.state_dict(),
                            }, savename)       
                    if valid_loss < best_loss or np.isnan(best_loss):
                        shutil.copy(savename, savename.replace('last.pt', 'best.pt'))
                        best_loss = valid_loss    
                        print('best loss updated to {:.5f}'.format(best_loss))
                        scheduler.step(valid_loss)    
        
if(__name__ == '__main__'):
    train()