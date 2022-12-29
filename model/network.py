from os import wait
import torch
import torch.nn as nn

from .parts import BaseNN, DenseBlock, TransitionBlock, CoverageAttention, Maxout
from logger import variable_logger as logger
log = logger(is_active=True)
log.is_active = False

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
            DenseBlock(num_feature, growth_rate, num_bn // 2, dropout_rate)
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
                 embedding_dim:int = 256, attention_size:int = 512,  # e, n_prim
                 device:str='cpu'):
        """
        y_{t-1}: [B, 1] --e--> embedded: [B,1,e] --┐
                                s_{t-1}: [1,B,n] --G1--> \\hat{s_t}: [B,1,n]
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
        C_prim, W_B, H_B = high_res_shape
        log.log("A.shape", low_res_shape)
        log.log("B.shape", high_res_shape)
        e, n, n_prim = embedding_dim, hidden_size, attention_size
        print("n, n' = ", n, n_prim)
        self.n = n

        self.embedding = nn.Embedding(len_word_list, embedding_dim)
        self.gru1 = nn.GRU(e, n, batch_first=True)
        self.gru2 = nn.GRU(C+C_prim, n, batch_first=True)
        self.W_nn_prim = nn.Linear(n, n_prim, bias=False)

        q = 256
        L = W * H
        L_prim = W_B * H_B
        log.log("L", L)
        log.log("L_prim", L_prim)
        self.att1 = CoverageAttention(C, q, n_prim, L, 11, 5, device)
        self.att2 = CoverageAttention(C_prim, q, n_prim, L_prim, 7, 3, device)

        self.W_s = nn.Linear(n, e, bias=False)
        self.W_c = nn.Linear(C+C_prim, e, bias=False)
        self.W_o = nn.Linear(e // 2, len_word_list, bias=False)
        self.maxout = Maxout(2)

        self.device = device

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.n))
    def reset(self, batch_size):
        self.att1.reset_alpha(batch_size)
        self.att2.reset_alpha(batch_size)

    def forward(self, yt_prev, st_prev, low_res, high_res) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(yt_prev)
        hat_s_t, _ = self.gru1(embedded, st_prev)
        hat_s_t_prim = self.W_nn_prim(hat_s_t.squeeze(1))
        cA_t = self.att1(low_res, hat_s_t_prim)
        cB_t = self.att2(high_res, hat_s_t_prim)
        c_t = torch.cat((cA_t, cB_t), dim=1)
        s_t, _ = self.gru2(c_t.unsqueeze(1), hat_s_t.permute(1, 0, 2)) # B, 1, n -> 1, B, n
        o = self.W_s(s_t) + self.W_c(c_t.unsqueeze(1)) + embedded
        o = self.maxout(o)
        o = self.W_o(o).squeeze(1)
        return o, s_t.permute(1, 0, 2) # [B, e], [1, B, n]

class Auto_E(BaseNN):
    def __init__(self, in_channel:int=1, oc1:int=684, oc2:int=792):
        super().__init__()                                #   C,   W,   H
        self.conv1 = nn.Conv2d(in_channel,  64, 3, 2, 1)  #   1, 512, 128 ->  64, 256, 64   1/2
        self.conv2 = nn.Conv2d(        64, 128, 3, 2, 1)  #  64, 256,  64 -> 128, 128, 32   1/4
        self.conv3 = nn.Conv2d(       128, 256, 3, 2, 1)  # 128, 128,  32 -> 256,  64, 16   1/8
        self.conv4 = nn.Conv2d(       256, 512, 3, 2, 1)  # 256,  64,  16 -> 512,  32,  8   1/16

        self.conv_re1 = nn.Conv2d(    256, oc1, 3, 1, 1)
        self.conv_re2 = nn.Conv2d(    512, oc2, 3, 1, 1)
    def forward(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(i)
        x = self.conv2(x)
        o1 = self.conv3(x)
        o2 = self.conv_re2(self.conv4(o1))
        o1 = self.conv_re1(o1)
        return (o1, o2)

class Auto_D(BaseNN):
    def __init__(self, oc1:int=684, oc2:int=792):
        super().__init__()
        self.deconv_re1 = nn.ConvTranspose2d(oc1, 256, 3, 1, 1)
        self.deconv_re2 = nn.ConvTranspose2d(oc2, 512, 3, 1, 1)
        self.deconv4 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,  64, 3, 2, 1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d( 64,   1, 3, 2, 1, output_padding=1)
    def forward(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.deconv_re1(i[0])
        x2 = self.deconv_re2(i[1])
        x = self.deconv4(x2)
        delta = x - x1
        x = self.deconv3(x)  # x & x1 should be similar
        x = self.deconv2(x)
        x = self.deconv1(x)
        return (x, delta)
