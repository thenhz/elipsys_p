import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NHz5(nn.Module):
    def __init__(self, shape, vocab_size):
        super(NHz5, self).__init__()
        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        
        self.conv3 = nn.Conv3d(256, 75, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM(6375, 128, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc = nn.Linear(256, vocab_size+1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        #TODO:check this flattening
        #batch_size, frames, height, width, channels  = x.size()
        #x = x.view(batch_size, frames, -1)  # Flatten the spatial dimensions
        x = x.view(x.size(0),x.size(1), -1)
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x = self.fc(x)
        x = self.softmax(x)
        x = x.permute(1,0,2).contiguous()
        
        return x