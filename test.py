import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from logger import variable_logger
from model.network import Encoder, Decoder
from dataloader.dataloader import dataset, collate_batch, START, PAD

device = "cpu"
path = "./data/imgs/converted/"
batch_size = 4

ds = dataset(path, tokens_file="./data/tokens.tsv", gt_file="./data/data.txt")
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
len_word_list = len(ds.token_to_id)

input_size = (200, 60)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)


enc = Encoder()
dec = Decoder(len_word_list, low_res_shape, high_res_shape)

for idx, data in enumerate(train_dl):
    img = data["img"]
    res = data["truth"]
    A, B = enc(img)
    print(A.shape)
    print(B.shape)
    expected = res["encoded"].to(device)
    expected[expected == -1] = train_dl.dataset.token_to_id[PAD]
    previous = expected[:, 0] # on all batch
    previous = previous.view(-1, 1)
    hidden = dec.init_hidden(batch_size)
    out, hidden = dec(previous, hidden, A, B)
    print(out.shape, hidden.shape)
    exit(0)
