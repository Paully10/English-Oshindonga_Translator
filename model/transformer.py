import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class AURATransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=128,
        pad_id=0
    ):

        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        # Token embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id
        )

        # Positional information
        self.positional = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Convert hidden states into vocabulary probabilities/logits
        self.output = nn.Linear(
            d_model,
            vocab_size
        )

    def generate_causal_mask(
        self,
        size,
        device
    ):

        return torch.triu(
            torch.ones(
                size,
                size,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

    def forward(
        self,
        src,
        tgt
    ):

        # Padding masks
        src_padding_mask = src.eq(
            self.pad_id
        )

        tgt_padding_mask = tgt.eq(
            self.pad_id
        )

        # Prevent decoder from seeing future tokens
        tgt_mask = self.generate_causal_mask(
            tgt.size(1),
            tgt.device
        )

        # Embeddings + positional encoding
        src_emb = self.positional(
            self.embedding(src)
            * math.sqrt(self.d_model)
        )

        tgt_emb = self.positional(
            self.embedding(tgt)
            * math.sqrt(self.d_model)
        )

        # Transformer
        hidden = self.transformer(
            src_emb,
            tgt_emb,

            tgt_mask=tgt_mask,

            src_key_padding_mask=src_padding_mask,

            tgt_key_padding_mask=tgt_padding_mask,

            memory_key_padding_mask=src_padding_mask
        )

        # Vocabulary logits
        logits = self.output(hidden)

        return logits