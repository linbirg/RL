"""
This post presents an annotated version of the paper in the form of a line-by-line implementation. 
It reorders and deletes some sections from the original paper and adds comments throughout. 
"""
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
# from torchtext.utils import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
# import spacy
# import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):

    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:

    def step(self):
        None


class EncoderDecoder(nn.Module):
    """
    标准的编解码器模型架构.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked source/target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask,
                            tgt_mask)  # memory: Encoder


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(model, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])  # 创建N个模块实


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 创建N个模块实例, 并将其传递给
        self.norm = LayerNorm(layer.size)  # 定义Layer Norm 模块。

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:  # 对每个Layer进行操作, 并将其传递给下一个
            x = layer(x, mask)  # 层。将其作为参数传递给下一个层。

        return self.norm(x)  # 输出应为Layer Norm中的输出.


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()  # 在构造函数中执行实现
        self.a_2 = nn.Parameter(
            torch.ones(features))  # Learnable parameter 1.0.1.1.1.1
        self.b_2 = nn.Parameter(
            torch.zeros(features))  # Learnable parameter 0.0.1.1
        self.eps = eps  #

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):  # 定义Sublayer Connection module.
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):  # 构造函数. 实现
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout),
                               2)  # 定义子句子重叠连接模块.
        self.size = size  # 定义总大小.

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):  # 构造函数. 实现N个子句子重叠加权加
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)  # 定义N个子句子重叠加权
        self.norm = LayerNorm(layer.size)  # 定义子句子总大小限制. 实现LayerNorm.

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:  # 对每个子句子进行一个加权和的重叠加权
            x = layer(x, memory, src_mask, tgt_mask)  # 调用子句子的forward函数. 实现加权和.

        return self.norm(x)  # 返回加权后的子句子总大小. 实现LayerNorm.


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward,
                 dropout):  # 构造函
        super(DecoderLayer, self).__init__()  # 调用父类的构造函数. 实现子句子
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout),
                               3)  # 定义四个子句子连接器. 实

    def forward(self, x, memory, src_mask, tgt_mask):
        # m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](
            x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)  # 调用子句子的forward函数.


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)  # (1, size, size) 一个向量.
    subsequent_mask = torch.triu(torch.ones(attn_shape),
                                 diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# def example_mask():
#     LS_data = pd.concat([
#         pd.DataFrame({
#             "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
#             "Window": y,
#             "Masking": x,
#         }) for y in range(20) for x in range(20)
#     ])

#     alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(
#         alt.X("Window:O"),
#         alt.Y("Masking:O"),
#         alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
#     ).interactive()

# execute_example(example_mask)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # (1, size, size)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. 
    In this work we employ h=8 parallel attention layers.
    """

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0  # make sure d_model is a multiple of h.
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # d_k is identical for all h.
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model),
                              4)  # 4 copies of the same linear.
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # (size, 1, size, size)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                 self.h * self.d_k))
        # ? lin 为何要删除？函数内删除外参数并不是一个好的做法
        del query
        del key
        del value

        return self.linears[-1](x)  # (size, 1, size, size)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (size, size, dim)
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embedings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


# def example_positional():
#     pe = PositionalEncoding(20, 0)
#     y = pe.forward(torch.zeros(1, 100, 20))

#     data = pd.concat(
#         [
#             pd.DataFrame(
#                 {
#                     "embedding": y[0, :, dim],
#                     "dimension": dim,
#                     "position": list(range(100)),
#                 }
#             )
#             for dim in [4, 5, 6, 7]
#         ]
#     )

#     return (
#         alt.Chart(data)
#         .mark_line()
#         .properties(width=800)
#         .encode(x="position", y="embedding", color="dimension:N")
#         .interactive()
#     )

# show_example(example_positional)


def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    dc = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, dc(attn), dc(ff), dropout), N),
        Decoder(DecoderLayer(d_model, dc(attn), dc(attn), dc(ff), dropout), N),
        nn.Sequential(Embedings(d_model, src_vocab), dc(position)),
        nn.Sequential(Embedings(d_model, tgt_vocab), dc(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():  # p.data.uniform_() 创建四个参数的nolinear函数
        if p.dim() > 1:  # 如果参数的维度大于1，则将其转换为向量形式。
            nn.init.xavier_uniform_(p)

    return model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(memory, src_mask, ys,
                                subsequent_mask(ys.size(1)).type_as(src.data))
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)