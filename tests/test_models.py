import pytest
import torch
from src.models import LSTMClassifier, TransformerClassifier

def test_lstm_forward():
    batch_size = 8
    seq_len = 20
    input_dim = 15
    num_classes = 3
    
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_size=32,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)
    
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()

def test_transformer_forward():
    batch_size = 4
    seq_len = 20
    input_dim = 12
    num_classes = 3
    
    model = TransformerClassifier(
        input_dim=input_dim,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        num_classes=num_classes,
        seq_len=seq_len,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)
    
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()
