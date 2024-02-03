import torch
from torch import nn
from typing import Tuple

class PositionalEmbedding(nn.Module):
    """Creates embedding + positional embeddings for a sentence.
    """

    def __init__(
            self,
            sentence_length: int,
            vocab_size: int,
            output_dim: int
        ):
        """Initializes layer.

        Args:
            sentence_length: Sentence length for positional embeddings.
            vocab_size: Size of vocabular of index vectors.
            output_dim: Dimension of embeddings.
        """
        super().__init__()

        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.token_embeddings = nn.Embedding(vocab_size, output_dim)
        self.position_embeddings = nn.Embedding(sentence_length, output_dim)

        self.positions = nn.Parameter(torch.arange(0, self.sentence_length, dtype = torch.int), requires_grad = False)

    def forward(
            self,
            x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes embeddings.

        Args:
            x: Batch of index vectors.

        Returns:
            Embeddings and their Masks (for attention).
        """
        mask = ~torch.not_equal(x, 0)
        return self.token_embeddings(x) + self.position_embeddings(self.positions), mask
        
class TransformerEncoder(nn.Module):
    """Simple encoder.
    """

    def __init__(
            self,
            embed_dim: int,
            ff_dim: int,
            num_heads: int
        ):
        """Initializes layer.
        
        Args:
            embed_dim: Dimension of embeddings.
            ff_dim: Neurons of feed_forward layer.
            num_heads: Number of attention heads.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first = True)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = ff_dim),
            nn.ReLU(),
            nn.Linear(in_features = ff_dim, out_features = embed_dim)
        )
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None
        ) -> torch.Tensor:
        """Computes encoded embeddings.
        """
        norm_att_out = self.norm_1(self.attention(x,x,x, need_weights = False, key_padding_mask = mask)[0] + x)
        norm_ff_out = self.norm_2(self.feed_forward(norm_att_out) + norm_att_out)
        return norm_ff_out