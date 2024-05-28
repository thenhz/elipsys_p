
import torch
import torch
import torch.nn as nn
import torchvision.models as models

class NHz7(nn.Module):
    def __init__(self, vocab_size, hidden_size=1024, num_layers=3, pretrained_model='resnet18'):
        """
        Initializes the model.

        Args:
        - vocab_size (int): Number of possible output tokens (size of the vocabulary).
        - hidden_size (int): Number of features in the hidden state of the recurrent layers.
        - num_layers (int): Number of recurrent layers (LSTM/GRU).
        - pretrained_model (str): The name of the pretrained model to use for feature extraction.
        """
        super(NHz7, self).__init__()

        # Pretrained feature extractor
        if pretrained_model == 'resnet18':
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor.fc = nn.Identity()  # Remove the last fully connected layer
            feature_dim = 512  # Dimension of the output features from ResNet
        elif pretrained_model == 'efficientnet_v2':
            self.feature_extractor = models.efficientnet_v2_s(pretrained=True)
            self.feature_extractor.classifier = nn.Identity()  # Remove the last fully connected layer
            feature_dim = 1280  # Dimension of the output features from EfficientNet V2
        else:
            raise NotImplementedError(f'{pretrained_model} is not implemented. Please use resnet18 or efficientnet_v2 for now.')

        # Recurrent layers
        #self.rnn = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.rnn = nn.GRU(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=True)

        # Classification layer
        self.fc = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, x, input_len):
        """
        Forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width).

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, num_frames, vocab_size).
        """
        batch_size, num_frames, channels, height, width = x.shape

        # Extract features from each frame
        features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]  # Extract the i-th frame
            frame_features = self.feature_extractor(frame)  # Shape: (batch_size, feature_dim)
            features.append(frame_features)

        features = torch.stack(features, dim=1)  # Shape: (batch_size, num_frames, feature_dim)

        # Process the sequence of features with the RNN
        rnn_out, _ = self.rnn(features)  # rnn_out shape: (batch_size, num_frames, hidden_size)

        # Classification for each timestep
        out = self.fc(rnn_out)  # Shape: (batch_size, num_frames, vocab_size)

        return out

# Example usage:
if __name__ == '__main__':
    # Sample input: 2 videos, each with 10 frames, 3 channels (RGB), 224x224 resolution
    sample_input = torch.randn(2, 10, 3, 224, 224)

    # Assuming vocab_size is 100 (for example, if we are predicting 100 different characters)
    vocab_size = 100
    model = NHz7(vocab_size=vocab_size, hidden_size=512, num_layers=2, pretrained_model='resnet18')
    output = model(sample_input)

    print(output.shape)  # Output: torch.Size([2, 10, 100])
