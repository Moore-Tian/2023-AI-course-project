from layers_of_transformer import *
import torch.nn as nn
import torch


# a simple version, which only has an encoder stack
class Transformer(nn.Module):
    def __init__(self, dim_embed, vocab_size, max_len, num_encoder, num_heads, dim_ff, dim_out, dropout):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(dim_embed, vocab_size)
        self.positional_encoding = positional_encoding(dim_embed, dropout, max_len)
        self.encoder_stack = encoder_stack(num_encoder, dim_embed, num_heads, dim_ff, dropout)
        self.linear = nn.Linear(dim_embed, dim_out)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder_stack(x, mask)
        x = self.linear(x)
        return x

    def predict(self, x, mask):
        scores = self.forward(x, mask)
        tags_pred = torch.argmax(scores, dim=-1)
        return tags_pred
