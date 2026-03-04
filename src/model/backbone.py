import torch
import torch.nn as nn

class LSTMBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        # input_dim = number input channels
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        # We only care about the output at the final timestep for forecasting
        final_feature_vector = lstm_out[:, -1, :] 
        return final_feature_vector
    
class SimpleTCNBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        # PyTorch Conv1d expects shape (batch, channels, length)
        # So we use input_dim as the in_channels
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, 
                               kernel_size=kernel_size, padding='same', dilation=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, 
                               kernel_size=kernel_size, padding='same', dilation=2)
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, input_dim)
        # Conv1d expects: (batch_size, input_dim, sequence_length)
        x = x.permute(0, 2, 1) 
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Take the features from the final time step
        final_feature_vector = x[:, :, -1] 
        return final_feature_vector