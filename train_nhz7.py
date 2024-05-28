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


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
        experiment
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.experiment = experiment
        self.scaler = GradScaler()
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # 0 is conventionally used for blank class in CTC
        self.running_loss = 0.0

    def _run_batch(self, inputs, input_len, labels, label_lengths):
        self.optimizer.zero_grad()
        with autocast():
            output = self.model(inputs, input_len)
            output_reshaped = torch.permute(output, (1, 0, 2))
            loss = self.loss_fn(output_reshaped.log_softmax(2), labels, input_len, label_lengths)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.running_loss += loss.item()
        return loss.item()

    def _run_epoch(self, epoch):
        self.running_loss = 0.0
        self.model.train(True)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        #for source, targets in self.train_data:
        for i, data in enumerate(self.train_data):
            step_num = (epoch+1)*i
            # Get inputs and labels
            inputs = data['video'].to(self.gpu_id)
            input_len = data['input_len'].to(self.gpu_id)
            labels = data['label'].to(self.gpu_id)
            label_lengths = data['output_len'].to(self.gpu_id)
            batch_loss = self._run_batch(inputs, input_len, labels, label_lengths)
            #last_loss = self.running_loss / (i + 1)  # average loss per batch
            self.experiment.log_metric("batch_loss", batch_loss, step=step_num)
            print(f"Train batch {i}, step {step_num}, loss {batch_loss}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        # Watch the model with Comet.ml
        with self.experiment.train():
            watch(self.model, log_step_interval=10)
            for epoch in range(max_epochs):
                self._run_epoch(epoch)
                #TODO: validation
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)


def load_train_objs(args):
    train_set = DummyDataset(args, "train")  # load your dataset
    model = model = NHz7(args['vocab_size'], hidden_size=1024, num_layers=3, pretrained_model=args['model_name']).to(device) # load your model
    lr = args['batch'] / 32.0 / torch.cuda.device_count() * args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    return train_set, model, optimizer, scheduler


def prepare_dataloader(dataset: Dataset, args):
    return DataLoader(
        dataset,
        batch_size=args['batch'],
        pin_memory=True,
        shuffle=True,
        num_workers=args['num_workers']
    )


def main(device, experiment, args):
    total_epochs = args['max_epoch']
    save_every = args['save_every']
    dataset, model, optimizer, scheduler = load_train_objs(args)
    train_data = prepare_dataloader(dataset, args)
    trainer = Trainer(model, train_data, optimizer, device, save_every, experiment)
    trainer.train(total_epochs)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    experiment = Experiment(
        api_key="8hE5H33Hi2gJSQMZF29RcRB3M",
        project_name="eLipSys_NHz7",
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
        auto_metric_step_rate=1
        )
    device = 0  # shorthand for cuda:0
    main(device, experiment, args)