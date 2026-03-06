import torch
import torch.nn as nn


class EMGScalogramBackbone(nn.Module):
    """
    Backbone for processing EMG scalogram data (time-frequency representations).
    Input shape: (batch, n_channels, time, freq_scales)
    """
    def __init__(self, n_channels, n_freq_scales, hidden_dim, backbone_type='conv2d_lstm'):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq_scales = n_freq_scales
        self.hidden_dim = hidden_dim
        self.backbone_type = backbone_type
        
        if backbone_type == 'conv2d_lstm':
            # 2D CNN to extract spatial-temporal-frequency features
            self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((16, 8))  # Reduce spatial dimensions
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
            # LSTM to process temporal sequence
            self.lstm = nn.LSTM(
                input_size=64 * 8,  # Flattened freq dimension
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            
        elif backbone_type == 'flatten_lstm':
            # Simple approach: flatten freq dimension and treat as features
            self.lstm = nn.LSTM(
                input_size=n_channels * n_freq_scales,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            
        else:
            raise ValueError(f"Unknown EMG backbone type: {backbone_type}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, time, freq_scales)
        Returns:
            features: (batch, hidden_dim)
        """
        batch_size, n_channels, time_steps, freq_scales = x.shape
        
        if self.backbone_type == 'conv2d_lstm':
            # Apply 2D convolutions
            x = self.relu(self.conv1(x))  # (batch, 32, time, freq)
            x = self.relu(self.conv2(x))  # (batch, 64, time, freq)
            x = self.pool(x)  # (batch, 64, 16, 8)
            x = self.dropout(x)
            
            # Reshape for LSTM: (batch, time, features)
            x = x.permute(0, 2, 1, 3)  # (batch, 16, 64, 8)
            x = x.reshape(batch_size, 16, -1)  # (batch, 16, 512)
            
            # LSTM processing
            lstm_out, _ = self.lstm(x)
            features = lstm_out[:, -1, :]  # Take last timestep
            
        elif self.backbone_type == 'flatten_lstm':
            # Flatten channels and freq dimensions
            x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
            x = x.reshape(batch_size, time_steps, -1)  # (batch, time, channels*freq)
            
            # LSTM processing
            lstm_out, _ = self.lstm(x)
            features = lstm_out[:, -1, :]  # Take last timestep
        
        return features


class AngleHistoryBackbone(nn.Module):
    """
    Backbone for processing angle history data (simple 1D temporal sequences).
    Input shape: (batch, n_angles, time)
    """
    def __init__(self, n_angles, hidden_dim, backbone_type='lstm'):
        super().__init__()
        self.n_angles = n_angles
        self.hidden_dim = hidden_dim
        self.backbone_type = backbone_type
        
        if backbone_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=n_angles,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
        elif backbone_type == 'tcn':
            self.conv1 = nn.Conv1d(n_angles, hidden_dim, kernel_size=3, padding='same', dilation=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding='same', dilation=2)
            self.relu = nn.ReLU()
        else:
            raise ValueError(f"Unknown angle backbone type: {backbone_type}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_angles, time)
        Returns:
            features: (batch, hidden_dim)
        """
        if self.backbone_type == 'lstm':
            # Permute to (batch, time, n_angles)
            x = x.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)
            features = lstm_out[:, -1, :]  # Take last timestep
            
        elif self.backbone_type == 'tcn':
            # x is already (batch, n_angles, time) for Conv1d
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            features = x[:, :, -1]  # Take last timestep
        
        return features


class DualBackbone(nn.Module):
    """
    Dual-backbone architecture that processes EMG and angle data separately,
    then fuses them. Supports different feature modes.
    """
    def __init__(self, emg_backbone, angle_backbone, feature_mode='both', fusion_hidden_dim=128):
        """
        Args:
            emg_backbone: Backbone for EMG scalogram processing
            angle_backbone: Backbone for angle history processing
            feature_mode: 'both', 'emg_only', or 'angle_only'
            fusion_hidden_dim: Hidden dimension for fusion layer
        """
        super().__init__()
        self.emg_backbone = emg_backbone
        self.angle_backbone = angle_backbone
        self.feature_mode = feature_mode
        
        # Calculate combined feature dimension
        if feature_mode == 'both':
            combined_dim = emg_backbone.hidden_dim + angle_backbone.hidden_dim
        elif feature_mode == 'emg_only':
            combined_dim = emg_backbone.hidden_dim
        elif feature_mode == 'angle_only':
            combined_dim = angle_backbone.hidden_dim
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hidden_dim = fusion_hidden_dim
    
    def forward(self, emg_data=None, angle_data=None):
        """
        Args:
            emg_data: (batch, n_channels, time, freq_scales) or None
            angle_data: (batch, n_angles, time) or None
        Returns:
            features: (batch, hidden_dim)
        """
        if self.feature_mode == 'both':
            if emg_data is None or angle_data is None:
                raise ValueError("Both EMG and angle data required for feature_mode='both'")
            emg_features = self.emg_backbone(emg_data)
            angle_features = self.angle_backbone(angle_data)
            combined_features = torch.cat([emg_features, angle_features], dim=1)
            
        elif self.feature_mode == 'emg_only':
            if emg_data is None:
                raise ValueError("EMG data required for feature_mode='emg_only'")
            combined_features = self.emg_backbone(emg_data)
            
        elif self.feature_mode == 'angle_only':
            if angle_data is None:
                raise ValueError("Angle data required for feature_mode='angle_only'")
            combined_features = self.angle_backbone(angle_data)
        
        # Fusion
        features = self.fusion(combined_features)
        return features