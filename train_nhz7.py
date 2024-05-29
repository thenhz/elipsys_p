# Import necessary modules from Comet.ml
from comet_ml import Experiment, init
from comet_ml.integration.pytorch import log_model, watch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import logging

def setup_logging(rank):
    logger = logging.getLogger(f'Process_{rank}')
    logger.setLevel(logging.INFO)

    # Create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(f'%(asctime)s [%(levelname)s] [Process {rank}] %(message)s',  
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Optionally, add a file handler as well
    file_handler = logging.FileHandler(f'./logs/process_{rank}.logs')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: int,
        save_every: int, 
        experiment, 
        num_to_char,
        scheduler,
        logger
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.model = DDP(model, device_ids=[device])
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.experiment = experiment
        self.scaler = GradScaler()
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # 0 is conventionally used for blank class in CTC
        self.running_loss = 0.0
        self.num_to_char = num_to_char
        self.scheduler = scheduler
        self.best_vloss = 1_000_000.
        self.logger = logger

    def _run_batch(self, inputs, input_len, labels, label_lengths, train=True):
        if train:
            self.optimizer.zero_grad()
        with autocast():
            output = self.model(inputs, input_len)
            output_reshaped = torch.permute(output, (1, 0, 2))
            loss = self.loss_fn(output_reshaped.log_softmax(2), labels, input_len, label_lengths)
        if train:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss.item(), output

    def _run_epoch(self, epoch):
        self.running_loss = 0.0
        self.model.train(True)
        batch_size = len(self.train_data)
        self.logger.info(f"[Device {self.device}] Epoch {epoch} | Total steps: {batch_size}")
        self.train_data.sampler.set_epoch(epoch)
        #for source, targets in self.train_data:
        for i, data in enumerate(self.train_data):
            step_num = (epoch*batch_size) + (i+1)
            # Get inputs and labels
            inputs = data['video'].to(self.device)
            input_len = data['input_len'].to(self.device)
            labels = data['label'].to(self.device)
            label_lengths = data['output_len'].to(self.device)
            batch_loss, output = self._run_batch(inputs, input_len, labels, label_lengths)
            self.running_loss += batch_loss
            #last_loss = self.running_loss / (i + 1)  # average loss per batch
            self.experiment.log_metric("loss", batch_loss, step=step_num)
            self.logger.info(f"    Train batch {i}, step {step_num}, loss {batch_loss}")

    def _validate(self, epoch):
        vloss = 0
        steps = 0
        with torch.no_grad():
            self.model.eval()
            batch_size = len(self.valid_data)
            for i, data in enumerate(self.valid_data):
                steps += 1
                step_num = (epoch*batch_size) + (i+1)
                inputs = data['video'].to(self.device)
                input_len = data['input_len'].to(self.device)
                labels = data['label'].to(self.device)
                label_lengths = data['output_len'].to(self.device)
                loss, output = self._run_batch(inputs, input_len, labels, label_lengths, train=False)
                self.experiment.log_metric("loss", loss, step=step_num)
                vloss += loss
                self.logger.info(f"    Valid batch {i}, step {i+1}, loss {loss}")
                decoded_truth = decode_predictions(torch.permute(labels.cpu(), (1, 0)), self.num_to_char)
                decoded_predictions = greedy_ctc_decode(output, input_len, self.num_to_char)

                # Log the predictions to Comet.ml
                for j, decoded in enumerate(decoded_predictions):
                    # Log the predicted and true values
                    self.experiment.log_text(
                        decoded, 
                        step=step_num, 
                        metadata={"type": "prediction", "batch": i, "item": j}
                        )
                    self.experiment.log_text(
                        decoded_truth[j], 
                        step=step_num, 
                        metadata={"type": "truth", "batch": i, "item": j}
                        )
                    self.logger.info(f"Epoch {epoch}, Item {j}:\nPrediction: {decoded}\nTruth: {decoded_truth[j]}")
                    if j == 1:
                        break
            return vloss / steps
                

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        model_path = 'elipsys_best'
        model_path = os.path.join("checkpoint", model_path)
        torch.save(ckp, model_path)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {model_path}")

    def train(self, max_epochs: int):        
        watch(self.model, log_step_interval=10)
        for epoch in range(max_epochs):
            with self.experiment.train():
                self._run_epoch(epoch)
            with self.experiment.validate():
                vloss = self._validate(epoch)
            self.scheduler.step(vloss)
            if vloss < self.best_vloss and self.device == 0:
                self._save_checkpoint(epoch)


def load_train_objs(args):
    char_to_num, num_to_char = word_to_number_mapping()
    train_set = DummyDataset(args, "train", char_to_num) 
    valid_set = DummyDataset(args, "val", char_to_num) # load your dataset
    model = model = NHz7(args['vocab_size'], hidden_size=1024, num_layers=3, pretrained_model=args['model_name'])
    lr = args['batch'] / 32.0 / torch.cuda.device_count() * args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    return train_set, valid_set, model, optimizer, scheduler, num_to_char


def prepare_dataloader(dataset: Dataset, args, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=args['batch'],
        pin_memory=True,
        shuffle=shuffle,
        num_workers=args['num_workers'],
        sampler=DistributedSampler(dataset)
    )

def word_to_number_mapping():
    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    char_to_num = {}
    num_to_char = {}
    for i, char in enumerate(vocab):
        char_to_num[char] = i
        num_to_char[i] = char
    return char_to_num, num_to_char

def decode_predictions(preds, num_to_char):
    """
    Decode numeric predictions into strings using the num_to_char mapping.

    Args:
    - preds (torch.Tensor): predicted numeric labels
    - num_to_char (dict): mapping from numeric labels to characters

    Returns:
    - decoded (list of str): decoded label strings
    """
    decoded_batch = []
    for i in range(preds.shape[1]):
        decoded = []
        for p in preds[:,i]:
            if int(p) in num_to_char:
                decoded.append(num_to_char[int(p)])
        decoded_batch.append(''.join(decoded))
    return decoded_batch

def greedy_ctc_decode(output, input_lengths, num_to_char):
    """
    Greedy CTC decode the output sequences.

    Args:
    - output (torch.Tensor): Output from the neural network of shape (batch_size, seq_len, vocab_size).
    - input_lengths (torch.Tensor): Lengths of the input sequences.
    - num_to_char (dict): Mapping from numeric labels to characters.

    Returns:
    - decoded (list of str): Decoded label strings.
    """
    decoded_batch = []
    output = output.softmax(2).cpu()
    for i in range(output.size(0)):
        probs = output[i, :input_lengths[i]]  # Get the sequence outputs for the given input length
        _, max_indices = torch.max(probs, dim=1)
        chars = [num_to_char[idx.item()] for idx in max_indices]
        decoded = []
        previous_char = None
        #TODO: perchè il num_to_char[0] è "a"? ho capito che facciamo padding così ma la "a" come la mappiamo?inoltre non docrei appendere 
        for char in chars:
            if char != previous_char and char != num_to_char[0]:  # skip blanks (conventionally mapped to 0)
                decoded.append(char)
            previous_char = char
        
        decoded_batch.append(''.join(decoded))

    return decoded_batch


def main(device, args, world_size):
    ddp_setup(device, world_size)
    logger = setup_logging(device)
    if device == 0:
        experiment = Experiment(
            api_key="8hE5H33Hi2gJSQMZF29RcRB3M",
            project_name="eLipSys_NHz7",
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
            auto_metric_step_rate=1
            )
    else:
        experiment = Experiment(disabled=True)
    experiment.log_parameters(args)
    total_epochs = args['max_epoch']
    save_every = args['save_every']
    train_ds, valid_ds,model, optimizer, scheduler, num_to_char = load_train_objs(args)
    train_data = prepare_dataloader(train_ds, args, shuffle=False)
    valid_data = prepare_dataloader(valid_ds, args, shuffle=False)
    trainer = Trainer(
        model, 
        train_data, 
        valid_data,
        optimizer, 
        device, 
        save_every, 
        experiment,
        num_to_char,
        scheduler,
        logger
    )
    trainer.train(total_epochs)
    log_model(experiment, model, "eLipSys_model")


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args, world_size), nprocs=world_size)