import math
import torch
import torch.nn as nn
from typing import Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        out = self.input_projection(x)  # shape: [batch_size, seq_len, hidden_size]
        out, (hn, cn) = self.lstm(out)  # out shape: [batch_size, seq_len, hidden_size]
        # Use the output of the last sequence step
        out = out[:, -1, :]  # shape: [batch_size, hidden_size]
        out = self.dropout(out)
        logits = self.classifier(out)  # shape: [batch_size, num_classes]
        return logits

class TransformerClassifier(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_heads: int, 
        num_layers: int, 
        dim_feedforward: int, 
        num_classes: int, 
        seq_len: int,
        dropout: float = 0.1,
        use_positional: bool = True
    ):
        super().__init__()
        self.use_positional = use_positional
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        if self.use_positional:
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len + 1, dropout=dropout)
            
        # CLS token similar to BERT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        
        # 1. Project input features to embedding dimension
        out = self.input_projection(x)  # shape: [batch_size, seq_len, embed_dim]
        
        # 2. Prepend CLS token to sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: [batch_size, 1, embed_dim]
        out = torch.cat((cls_tokens, out), dim=1)  # shape: [batch_size, seq_len + 1, embed_dim]
        
        # 3. Add positional encoding
        if self.use_positional:
            out = self.pos_encoder(out)
            
        # 4. Pass through Transformer Encoder
        out = self.transformer_encoder(out)  # shape: [batch_size, seq_len + 1, embed_dim]
        
        # 5. Extract representation of the CLS token
        cls_rep = out[:, 0, :]  # shape: [batch_size, embed_dim]
        
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep)  # shape: [batch_size, num_classes]
        return logits
