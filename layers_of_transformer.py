import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_attention


# 词嵌入层
class Embeddings(nn.Module):
    def __init__(self, dim_embed, vocab):
        super(Embeddings, self).__init__()
        self.dim_embed = dim_embed
        self.mapping = nn.Embedding(vocab, dim_embed)

    def forward(self, x):
        """ print('original shape =', x.shape)
        print('shape after embedding = ', self.mapping(x).shape) """
        return self.mapping(x) * math.sqrt(self.dim_embed)


# 位置编码层
class positional_encoding(nn.Module):
    def __init__(self, dim_embed, dropout, max_len):
        super(positional_encoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_len, dim_embed)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embed, 2) * -(math.log(10000.0) / dim_embed))

        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)

        """ self.pos_encoding = pos_encoding """
        # 这么高级的方法我还是之后再用吧
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        """ print('shape of pos_encoding = ', self.pos_encoding.shape) """
        x = x + self.pos_encoding[:, :x.shape[-2]]

        # 嘿，为什么要dropout
        return self.dropout(x)


# encoder和decoder会用到的多头注意力层
class multi_head_attention(nn.Module):
    def __init__(self, dim_embed, num_heads, dropout):
        super(multi_head_attention, self).__init__()
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.dim_per_head = dim_embed // num_heads

        self.linear_q = nn.Linear(dim_embed, dim_embed)
        self.linear_k = nn.Linear(dim_embed, dim_embed)
        self.linear_v = nn.Linear(dim_embed, dim_embed)
        self.linear_out = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, mask=None):
        """ print('mask.shape = ', mask.shape) """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = input.shape[0]

        query = self.linear_q(input).view(batch_size, -1, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        key = self.linear_k(input).view(batch_size, -1, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        value = self.linear_v(input).view(batch_size, -1, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        """ print('Q.shape = ', query.shape) """

        output, self.attention_weights = compute_attention(query, key, value, mask)
        """ print('out_put.shape = ', output.shape) """
        output = self.dropout(output)

        # 为什么要做这些奇奇怪怪的维度转换啊
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.dim_embed)

        return self.linear_out(output)


# 层归一化
class normalization(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(normalization, self).__init__()
        self.var = nn.Parameter(torch.ones(size))
        self.mean = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.var * (x - mean) / (std + self.eps) + self.mean


# 前馈层
class feed_forward(nn.Module):
    def __init__(self, dim_embed, dim_ff, dropout):
        super(feed_forward, self).__init__()
        self.linear1 = nn.Linear(dim_embed, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_embed)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# 单个encoder块
class encoder(nn.Module):
    def __init__(self, dim_embed, num_heads, dim_ff, dropout):
        super(encoder, self).__init__()
        self.muti_head_attention = multi_head_attention(dim_embed, num_heads, dropout)
        self.norm_1 = normalization(dim_embed)
        self.feed_forward = feed_forward(dim_embed, dim_ff, dropout)
        self.norm_2 = normalization(dim_embed)

    def forward(self, x, mask):
        """ print(x.shape) """
        x_atten = self.muti_head_attention(x, mask)
        """ print(x_atten.shape) """
        x = self.norm_1(x + x_atten)
        x_ff = self.feed_forward(x)
        x = self.norm_2(x + x_ff)
        return x


# encoder的堆栈
class encoder_stack(nn.Module):
    def __init__(self, num_layers, dim_embed, num_heads, dim_ff, dropout):
        super(encoder_stack, self).__init__()
        self.encoder_stack = nn.ModuleList([encoder(dim_embed, num_heads, dim_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for encoder in self.encoder_stack:
            x = encoder(x, mask)
        return x


# decoder内部的注意力层
class encoder_decoder_multi_head_attention(nn.Module):
    def __init__(self, dim_embed, num_heads, dropout):
        super(encoder_decoder_multi_head_attention, self).__init__()
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.dim_per_head = dim_embed // num_heads

        self.linear_q = nn.Linear(dim_embed, dim_embed)
        self.linear_k = nn.Linear(dim_embed, dim_embed)
        self.linear_v = nn.Linear(dim_embed, dim_embed)
        self.linear_out = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, decoder_input, encoder_input, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = input.shape[0]

        query = self.linear_q(decoder_input).view(batch_size, -1, self.num_heads, self.dim_per_head)
        key = self.linear_k(encoder_input).view(batch_size, -1, self.num_heads, self.dim_per_head)
        value = self.linear_v(encoder_input).view(batch_size, -1, self.num_heads, self.dim_per_head)

        output, self.attention_weights = compute_attention(query, key, value, mask)
        output = self.dropout(output)

        # 为什么要做这些奇奇怪怪的维度转换啊
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_embed)

        return self.linear_out(output)


# 单个decoder块
class decoder(nn.Module):
    def __init__(self, dim_embed, num_heads, dim_ff, dropout):
        super(decoder, self).__init__()
        self.multi_head_attention = multi_head_attention(dim_embed, num_heads, dropout)
        self.norm_1 = normalization(dim_embed)
        self.encoder_decoder_multi_head_attention = encoder_decoder_multi_head_attention(dim_embed, num_heads, dropout)
        self.norm_2 = normalization(dim_embed)
        self.feed_forward = feed_forward(dim_embed, dim_ff, dropout)
        self.norm_3 = normalization(dim_embed)

    def forward(self, decoder_input, encoder_input, mask):
        x_atten = self.multi_head_attention(decoder_input, mask)
        x = self.norm_1(x + x_atten)
        x_atten = self.encoder_decoder_multi_head_attention(x, decoder_input, encoder_input, mask)
        x = self.norm_2(x + x_atten)
        x_ff = self.feed_forward(x)
        x = self.norm_3(x + x_ff)
        return x


# decoder的堆栈
class decoder_stack(nn.Module):
    def __init__(self, num_layers, dim_embed, num_heads, dim_ff, dropout):
        super(decoder_stack, self).__init__()
        self.decoder_stack = nn.ModuleList([decoder(dim_embed, num_heads, dim_ff, dropout) for _ in range(num_layers)])

    def forward(self, decoder_input, encoder_input, mask):
        for decoder in self.decoder_stack:
            decoder_input = decoder(decoder_input, encoder_input, mask)
        return decoder_input


# 最后的结果生成
class genarator(nn.Module):
    def __init__(self, dim_embed, num_vocab):
        super(genarator, self).__init__()
        self.dim_embed = dim_embed
        self.vocab = vocab
        self.linear_out = nn.Linear(dim_embed, num_vocab)

    def forward(self, x):
        return F.log_softmax(self.linear_out(x), dim=-1)
