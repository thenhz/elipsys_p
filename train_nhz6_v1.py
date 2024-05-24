# Import necessary modules from Comet.ml
from comet_ml import Experiment, init
from comet_ml.integration.pytorch import log_model, watch

# Other necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from datetime import datetime
import os

from datasets.dummy_dataset import DummyDataset
from models.NHz7 import NHz7
from torch.cuda.amp import GradScaler, autocast

with open('config.yaml', 'r') as file:
    args = yaml.safe_load(file)

def train_one_epoch(epoch_index, experiment):
    running_loss = 0.0
    for i, data in enumerate(training_loader):
        # Get inputs and labels
        inputs = data['video'].to(device)
        input_len = data['input_len'].to(device)
        labels = data['label'].to(device)
        label_lengths = data['output_len'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        with autocast():
            outputs = model(inputs, input_len)
            outputs_reshaped = torch.permute(outputs, (1, 0, 2))
            loss = loss_fn(outputs_reshaped.log_softmax(2), labels, input_len, label_lengths)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss / (i + 1)  # average loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_index * len(training_loader) + i + 1
        # Log loss to Comet.ml
        experiment.log_metric("batch_loss", last_loss, step=tb_x)

    return last_loss

def get_data_loaders(args):
    #TODO: si può fare con un solo argomento
    train_ds = DummyDataset(args['video_dir'], args['label_dir'], "train", args)
    val_ds = DummyDataset(args['video_dir'], args['label_dir'], "val", args)
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

scaler = GradScaler()
# Initialize Comet.ml
#init()
experiment = Experiment(
    api_key=os.getenv("EXPERIMENT_API_KEY"),
    project_name="eLipSys_NHz7",
    #auto_histogram_weight_logging=True,
    #auto_histogram_gradient_logging=True,
    #auto_histogram_activation_logging=True
    )



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model = NHz7(args['vocab_size'], hidden_size=512, num_layers=1, pretrained_model=args['model_name']).to(device)
if args['gpus'] > 1:
    print("parallelling model...")
    model =  nn.DataParallel(model)

lr = args['batch'] / 32.0 / torch.cuda.device_count() * args['lr']
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # 0 is conventionally used for blank class in CTC
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Log hyperparameters to Comet.ml
experiment.log_parameters(args)

training_loader, validation_loader = get_data_loaders(args)

epoch_number = 0

EPOCHS = args['max_epoch']

best_vloss = 1_000_000.

# Watch the model with Comet.ml
with experiment.train():
    watch(model)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, experiment)

        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                videos = vdata['video'].to(device)
                input_len = vdata['input_len'].to(device)
                labels = vdata['label'].to(device)
                label_lengths = vdata['output_len'].to(device)

                with autocast():
                    voutputs = model(videos, input_len)
                    voutputs_reshaped = torch.permute(voutputs, (1, 0, 2))
                    vloss = loss_fn(voutputs_reshaped.log_softmax(2), labels, input_len, label_lengths)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log training and validation loss to Comet.ml
        experiment.log_metric("train_loss", avg_loss, epoch=epoch_number + 1)
        experiment.log_metric("valid_loss", avg_vloss, epoch=epoch_number + 1)

        # Get gradients
        gradients = {}
        for name, param in model.named_parameters():
            if "feature_extractor" not in name:
                if param.grad is not None:
                    gradients[name] = param.grad.detach()

        for name, gradient in gradients.items():
            experiment.log_histogram_3d(gradient.cpu().numpy(), name=f"gggradient_{name}", step=epoch_number + 1)

        # Log learning rate to Comet.ml
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            experiment.log_metric("learning_rate", lr, epoch=epoch_number + 1)
            break  # Assuming all param_groups have the same learning rate

        # Adjust learning rate based on validation loss
        scheduler.step(avg_vloss)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'elipsys_{}_{}'.format(timestamp, epoch_number)
            model_path = os.path.join(args['save_prefix'], model_path)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

# Log the model to Comet.ml for tracking and deployment
log_model(experiment, model, "eLipSys_model")
