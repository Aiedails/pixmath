import torch
import torch.nn as nn

from .parts import BaseNN, DenseBlock, TransitionBlock
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
