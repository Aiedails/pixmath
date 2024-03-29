import os
import torch
import torch.nn as nn

from logger import variable_logger as logger
log = logger(is_active=True)
log.is_active = False

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
    def auto_save(self, prefix:str, epoch: int, iter: int): # name-epoch-{epoch}-iter-{iter}
        save_name = prefix + "-epoch-" + str(epoch) + "-iter-" + str(iter)
        self.save(save_name)
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    def auto_load(self, ckpt_dir: str, prefix: str):
        self.ckpts = os.listdir(ckpt_dir)
        all_ckpt = []
        for ckpt_name in self.ckpts:
            ckpt_name = ckpt_name.split("-")
            if(ckpt_name[0] == prefix):
                all_ckpt.append( (ckpt_name, int(ckpt_name[2]), int(ckpt_name[4])) ) # name-epoch-{epoch}-iter-{iter}
        if (all_ckpt == []):
            print("no checkpoint founded.")
            return 0, 0
        res = sorted(all_ckpt, key = lambda x: (x[1], x[2]))
        newest = '-'.join(res[-1][0])
        print("loading newest checkpoints " + os.path.join(ckpt_dir, newest))
        self.load(os.path.join(ckpt_dir, newest))
        return int(res[-1][1]), int(res[-1][2])

class BottleneckBlock(BaseNN):
    """
    The Bottleneck Block. Each contains a 1x1 conv2d and a 3x3 one.
    in_channel -> 4 * grouth_rate -> cat[in_channel, grouth_rate]
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

class CoverageAttention(BaseNN):
    def __init__(self, a_channel, f_channel, hidden_dim, L, conv_kernel, conv_padding, device):
        """              |           |          |        |
                         C           q          n'       L
        A:          [B,C,L]
        \\hat{s_t}: [B,n']
        \\alpha:    [B,1,L] --reshape+conv--> F: [B,q,L]

        a_1: [L, 1] -- \\alpha_t1 
        a_2: [L, 1] -- \\alpha_t2 --> ca_t
        a_3: [L, 1] -- \\alpha_t3

        \\alpha_ti = softmax(e_ti)
        e_ti = v_att^T * (U_s * \\hat{s_t} +   U_a * a_i  +    U_f  * f_i  )
        [1,1]= [1, n'] * ([n',n] * [n,1]  + [n',C] * [C,1] + [n',q] * [q,1])
        => use the form X.dot(W). ignore all the transpose
        [L,1]= ([L,n]*[n,n'] + [L,C]*[C,n'] + [L,q]*[q,n']) * [n',1]
        """
        super().__init__()
        C, q, n_prim = a_channel, f_channel, hidden_dim
        self.q = q
        self.C = C
        self.n_prim = n_prim

        # origin implementation
        # self.U_a = nn.Parameter(torch.empty((hidden_dim, a_channel)))
        # self.U_f = nn.Parameter(torch.empty((hidden_dim, f_channel)))
        # self.U_s = nn.Parameter(torch.empty(hidden_dim))
        # nn.init.xavier_normal_(self.U_a)
        # nn.init.xavier_normal_(self.U_f)
        # nn.init.xavier_normal_(self.U_s.unsqueeze(0)) # need a 2D tensor.

        self.conv = nn.Conv2d(1, q, kernel_size=conv_kernel, padding=conv_padding)

        self.U_a = nn.Linear(C, n_prim, bias=False)
        self.U_f = nn.Linear(q, n_prim, bias=False)
        self.U_v = nn.Linear(n_prim, 1, bias=False)

        self.alpha = None
        self.device = device
        self.L = L

    def reset_alpha(self, batch_size):
        self.alpha = torch.zeros(batch_size, 1, self.L).to(self.device)
        log.log("alpha shape", self.alpha.shape)

    def forward(self, i, hat_s_t_prim_converted):
        B, C, H, W = i.shape
        if (self.alpha == None):
            self.reset_alpha(B)
        log.log("i.shape", i.shape)
        F = self.conv(self.alpha.reshape(B,1,H,W)).reshape(B, -1, H*W) # B, q, L
        log.log("F.shape", F.shape)
        assert(F.shape[1] == self.q)
        A = i.reshape(B, -1, H*W).permute(0, 2, 1) # B, C, L -> B, L, C
        # log.log("hat_s_t_prim_converted",hat_s_t_prim_converted.shape)
        log.log("A.shape", A.shape)
        # print(self.C, self.n_prim)
        res_a = self.U_a(A) # B, L, n'
        log.log("res_a.shape", res_a.shape)
        res_s = hat_s_t_prim_converted.unsqueeze(1).expand(B, self.L, self.n_prim) # B, n' -> B, L, n'
        F = F.permute(0, 2, 1) # B, q, L -> B, L, q
        res_f = self.U_f(F) # B, L, n'
        e = self.U_v( res_a + res_s + res_f ) # B, L, 1
        self.alpha = torch.softmax(e.permute(0, 2, 1), dim=1).detach() # B, 1, L
        c_At = (self.alpha * A.permute(0, 2, 1)).sum(2) # B, 1, L * B, C, L
        log.log("c_At.shape", c_At.shape)
        return c_At # B, L

class Maxout(BaseNN):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
    def forward(self, i):
        [*shape, last] = i.size()
        out = i.view(*shape, last // self.pool_size, self.pool_size)
        out, _ = out.max(-1)
        return out

