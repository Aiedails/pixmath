from os import wait
import torch
import torch.nn as nn

from .parts import BaseNN, DenseBlock, TransitionBlock, CoverageAttention, Maxout
from logger import variable_logger as logger
log = logger(is_active=True)

class Encoder(BaseNN):
    def __init__(self, img_channel:int=1, conv0_out_channel:int=48, num_bn:int=16,
                 growth_rate:int=24, dropout_rate:float=0.2):
        super().__init__()
        """
        img -> conv0 + norm0 -> block1
        --> trans1 -> block2
        --> trans2 -> block3_A -> out_A
         └> batchnorm + relu -> block3_B -> out_B
        """
        self.conv0 = nn.Conv2d(img_channel, conv0_out_channel, 7, 2, 3)
        self.norm0 = nn.BatchNorm2d(conv0_out_channel)
        num_feature = conv0_out_channel
        self.block1 = DenseBlock(num_feature, growth_rate, num_bn, dropout_rate)
        num_feature += num_bn * growth_rate
        self.trans1 = TransitionBlock(num_feature, 2) # channel cut by 2
        num_feature //= 2

        self.block2 = DenseBlock(num_feature, growth_rate, num_bn, dropout_rate)
        num_feature += num_bn * growth_rate

        self.block3_B = nn.Sequential(   # cause the B part does not down sampling,
            nn.BatchNorm2d(num_feature), # we manully add a batchnorm and relu here,
            nn.ReLU(inplace=True),       # together with a denseblock.
            DenseBlock(num_feature, growth_rate, num_bn, dropout_rate)
        )

        self.trans2 = TransitionBlock(num_feature, 2) # now we cut the channel for A part
        num_feature //= 2

        self.block3_A = DenseBlock(num_feature, growth_rate, num_bn, dropout_rate)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.conv0(i)
        o = self.relu(self.norm0(o))
        o = self.max_pool(o)

        o = self.block1(o) # C += num_bn * growth_rate
        o = self.trans1(o) # C, H, W //= 2

        o = self.block2(o) # C += num_bn * growth_rate

        o_B = self.block3_B(o)

        o_A = self.trans2(o) # C, H, W //= 2
        o_A = self.block3_A(o_A)
        return (o_A, o_B)

class Decoder(BaseNN):
    """
    ┌─ y_{t-1}─┐  ┌──┐         ┌─────────────────────────┐  ┌──┐
    │          ├─►│G1├─►\\hat{s_t} ─l─►                  ├─►│G2├──► s_t
    │  s_{t-1}─┘  └──┘           A ───► f_{catt} ─► c_t──┘  └──┘    │
    │                            B ───►              │  ┌───────────┘
    └─────────────────────────────────────────────┐E │l │l
                                                  ▼  ▼  ▼
                                                 maxout ──l──► softmax ──► y_t
    """
    def __init__(self, len_word_list:int, low_res_shape:tuple[int,int,int],
                 high_res_shape:tuple[int,int,int], hidden_size:int = 256, # n
                 embedding_dim:int = 256, attention_size:int = 512): # e, n_prim
        """
        y_{t-1}: [B, w] --e--> embedded: [B, e] --┐
                                s_{t-1}: [B, n] --G1--> \\hat{s_t}: [B, n]
                                 ┌---- \\hat{s_t}':[B,n'] <─┘l  |
             [B,C+C'] :c_t <--f_catt-- A,B: [B,L,C],[B,4L,C']   |
                        └─------------------------------------> G2 --> s_t:[B, n]
        s_t: [B, n]
        c_t: [B, C+C']
        emb: [B, e]
         s_t  *  W_s  +    c_t   * W_c      + emb
        [B,n] * [n,e] + [B,C+C'] * [C+C',e] + [B,e] = [B,e]

        """
        super().__init__()
        C,W,H = low_res_shape
        _,_,C_prim= high_res_shape
        e, n, n_prim = embedding_dim, hidden_size, attention_size

        self.embedding = nn.Embedding(len_word_list, embedding_dim)
        self.gru1 = nn.GRU(e, n, batch_first=True)
        self.gru2 = nn.GRU(C+C_prim, n, batch_first=True)
        self.W_s = nn.Linear(n, n_prim, bias=False)

        q = 256
        L = W * H
        self.att1 = CoverageAttention(C, q, n_prim, L, 11, 5)
        self.att2 = CoverageAttention(C_prim, q, n_prim, 4*L, 7, 3)

        self.W_s = nn.Linear(n, e, bias=False)
        self.W_c = nn.Linear(C+C_prim, e, bias=False)
        self.W_o = nn.Linear(e, len_word_list, bias=False)
        self.maxout = Maxout(2)

    def forward(self, yt_prev, st_prev, low_res, high_res) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(yt_prev)
        hat_s_t, _ = self.gru1(embedded, st_prev)
        hat_s_t_prim = self.W_s(hat_s_t)
        cA_t = self.att1(hat_s_t_prim, low_res)
        cB_t = self.att2(hat_s_t_prim, high_res)
        c_t = torch.cat((cA_t, cB_t), dim=1)
        s_t = self.gru2(c_t, hat_s_t)
        o = self.W_s(s_t) + self.W_c(c_t) + embedded
        o = self.maxout(o)
        o = self.W_o(o)
        return o, s_t
