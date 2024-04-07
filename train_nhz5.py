import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.NHz5 import NHz5
from datasets.dummy.LoadDummy import DummyDataset
import yaml

def train(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            #TODO: capire bene qui cosa sta facendo (occhio che se cambi il formato dei frame cambiano anche queste)
            outputs = model(inputs)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.int32)
            target_lengths = torch.full(size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.int32)
            loss = criterion(outputs, labels, input_lengths, target_lengths)

            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if batch_idx % 10 == 9:    # print every 10 mini-batches
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    print('Finished Training')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('config.yaml', 'r') as file:
        config_params = yaml.safe_load(file)
    # Initialize the dataset and dataloader
    dataset = DummyDataset(video_dir=config_params['video_dir'], label_dir=config_params['label_dir'])
    dataloader = DataLoader(dataset, batch_size=config_params['batch'], shuffle=True)

    # Initialize the model
    model = NHz5(config_params['shape'],config_params['vocab_size']).to(device)
    # to run in multi-gpu env
    #model = nn.DataParallel(model).to(device)

    # Define the loss function and optimizer
    criterion = nn.CTCLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    epochs = 10  # You can adjust the number of epochs
    train(model, dataloader, criterion, optimizer, epochs, device)

if __name__ == '__main__':
    main()
