import random
import torch
import math
import numpy as np
from NER_dataset import NER_dataset


# 输入语言类型，返回相应的训练集和测试集
def get_data_set(language):
    trainpath = f'./NER/{language}/train.txt'
    validpath = f'./NER/{language}/validation.txt'
    train_data = NER_dataset(trainpath)
    valid_data = NER_dataset(validpath)
    return train_data, valid_data


def batch_iter(data, batch_size=32, shuffle=True):
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags


def pad(data, padded_token, device):
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


# 用于transformer的多头注意力层
def compute_attention(Q, K, V, mask):
    d_k = Q.size()[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    """ print(scores.shape)
    print('mask.shape= ', mask.shape) """
    mask = mask.unsqueeze(1) * mask.unsqueeze(-1)
    """ print('mask.shape = ', mask.shape) """
    scores = scores.masked_fill(mask == 0, -1e16)
    attention_weights = torch.softmax(scores, dim=-1)
    attention_weights = attention_weights.masked_fill(mask == 0, 0)
    return torch.matmul(attention_weights, V), attention_weights


# 用于给transformer的decoder生成mask(不过仔细想想NER中好像用不到诶)
def decoder_self_atten_mask_genarate(size):
    mask = np.tril(np.ones((size, size)))
    mask[np.triu_indices(size, 1)] = 0
    return torch.from_numpy(mask)