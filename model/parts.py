import os
import torch
import torch.nn as nn

from logger import variable_logger as logger
log = logger(is_active=True)

class BaseNN(nn.Module):
    """
    Base NN, all other NN will base on this
    - __init__: virtual
    - forward(): virtual
    """

    def __init__(self):
        super().__init__()

    def forward(self, i: torch.Tensor): return i
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    def auto_load(self, ckpt_dir: str, prefix: str):
        self.ckpts = os.listdir(ckpt_dir)
        all_ckpt = []
        for ckpt_name in self.ckpts:
            ckpt_name = ckpt_name.split("-")
            if(ckpt_name[0] == prefix):
                all_ckpt.append( (ckpt_name, ckpt_name[2], ckpt_name[4]) ) # name-epoch-{epoch}-iter-{iter}
        res = sorted(all_ckpt, key = lambda x: (x[1], x[2]))
        newest = '-'.join(res[-1][0])
        print("loading newest checkpoints " + os.path.join(ckpt_dir, newest))
        self.load(os.path.join(ckpt_dir, newest))

class BottleneckBlock(BaseNN):
    """
    The Bottleneck Block. Each contains a 1x1 conv2d and a 3x3 one.
    in_channel -> 4 * grouth_rate -> grouth_rate
    """
    def __init__(self, in_channel: int, growth_rate: int, dropout_rate:float=0.2):
        super().__init__()
        inter_size = 4 * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, inter_size, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(inter_size, growth_rate, 3, 1, 1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, i: torch.Tensor) -> torch.Tensor: # remember the outputs are concatenated
        o = self.conv1(self.relu(self.norm1(i))) # origin implementation.
        o = self.conv2(self.relu(self.norm2(o)))
        # o = self.relu(self.norm1(self.conv1(i)))
        # o = self.relu(self.norm2(self.conv2(o)))
        o = self.dropout(o)
        return torch.cat([i, o], 1) # concat the return blocks

class DenseBlock(BaseNN):
    """
    The DenseB in the paper
    in_c --B-> in_c + growth_rate --B-> in_c + 2*gr --B--> ... --B-> in_c + num_bn*gr
    """
    def __init__(self, in_channel: int, growth_rate: int, num_bn: int, dropout_rate:float=0.2):
        super().__init__()
        layers = [
            BottleneckBlock(
                    in_channel + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            ) for i in range(num_bn)
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, i) -> torch.Tensor:
        return self.block(i)

class TransitionBlock(BaseNN):
    def __init__(self, in_channel:int, cut_rate:int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, in_channel // cut_rate, 1, 1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, i) -> torch.Tensor:
        o = self.conv(self.relu(self.norm(i)))
        o = self.pool(o)
        return o
