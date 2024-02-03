from .layers import *
from torch import nn
import torch
import torchtext

class TweetsDisasterClassifier(nn.Module):
    """Classifier based on simple encoder.
    """

    def __init__(
            self,
            sentence_length: int,
            vocab_size: int,
            embed_dim: int,
            ff_dim: int,
            num_attention_heads: int
        ):
        """Initializes model.
        """
        super().__init__()
        self.embeddings = PositionalEmbedding(sentence_length, vocab_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, ff_dim, num_attention_heads)
        self.max_pool = nn.MaxPool2d(kernel_size = (sentence_length,1))
        self.dropout = nn.Dropout()
        self.output = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = 1),
            nn.Sigmoid()
        )

    def forward(
            self,
            x: int
        ) -> torch.Tensor:
        embeddings, mask = self.embeddings(x)
        encoded = self.encoder(embeddings, mask)
        max_pooled = self.max_pool(encoded)
        return self.output(self.dropout(max_pooled))