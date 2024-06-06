
import torch
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

from datasets.facelandmark.mediapipe.blazeface_landmark import BlazeFaceLandmark

class NHz7(nn.Module):
    def __init__(self, vocab_size, hidden_size=1024, num_layers=3, feature_dim=1404):
        """
        Initializes the model.

        Args:
        - vocab_size (int): Number of possible output tokens (size of the vocabulary).
        - hidden_size (int): Number of features in the hidden state of the recurrent layers.
        - num_layers (int): Number of recurrent layers (LSTM/GRU).
        - pretrained_model (str): The name of the pretrained model to use for feature extraction.
        """
        super(NHz7, self).__init__()

        # First Recurrent layer
        self.rnn1 = nn.GRU(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        # Second Recurrent layer
        self.rnn2 = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        # Classification layer
        self.fc = nn.Linear(hidden_size*2, vocab_size)

        self.init_weights()

    def forward(self, x, input_len):
        """
        Forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width).

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, num_frames, vocab_size).
        """
        """ batch_size, num_frames, channels, height, width = x.shape

        # Extract features from each frame
        features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]  # Extract the i-th frame
            flags, frame_features = self.feature_extractor(frame)  # Shape: (batch_size, feature_dim)
            features.append(frame_features)

        features = torch.stack(features, dim=1)  # Shape: (batch_size, num_frames, feature_dim) """

        # Process the sequence of features with the first LSTM
        rnn_out1, _ = self.rnn1(x)  # rnn_out1 shape: (batch_size, num_frames, hidden_size)

        # Process the output of the first LSTM with the second LSTM
        rnn_out2, _ = self.rnn2(rnn_out1)  # rnn_out2 shape: (batch_size, num_frames, hidden_size)

        # Classification for each timestep
        out = self.fc(rnn_out2)  # Shape: (batch_size, num_frames, vocab_size)

        return out
    
    def init_weights(self):
        # Initialize LSTM weights for both LSTMs
        for name, param in self.rnn1.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for name, param in self.rnn2.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize Fully Connected Layer
        init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

# Example usage:
if __name__ == '__main__':
    # Sample input: 2 videos, each with 10 frames, 3 channels (RGB), 224x224 resolution
    sample_input = torch.randn(2, 10, 3, 224, 224)

    # Assuming vocab_size is 100 (for example, if we are predicting 100 different characters)
    vocab_size = 100
    model = NHz7(vocab_size=vocab_size, hidden_size=512, num_layers=2, pretrained_model='resnet18')
    output = model(sample_input)

    print(output.shape)  # Output: torch.Size([2, 10, 100])
