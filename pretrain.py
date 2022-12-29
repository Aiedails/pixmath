import os
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from logger import variable_logger, logger
from model.network import Encoder, Decoder, Auto_E, Auto_D
from dataloader.dataloader import dataset, collate_batch, START, PAD

writer = logger("runs/test")


device = "cpu" 
path = "./data/train/" 
ckpt_path = "./checkpoints/"
batch_size = 4
ds = dataset(path, tokens_file="./data/tokens.tsv", gt_file="./data/groundtruth_train.tsv", device=device)
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True)

enc = Auto_E().to(device)
dec = Auto_D().to(device)

certification = nn.L1Loss()
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)

with tqdm(total=len(train_dl)) as pbar:
    for i, data in enumerate(train_dl):

        img = data["image"].to(device)
        o1_o2 = enc(img)
        res, delta = dec(o1_o2)

        loss = certification(res, img) + delta.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(o1_o2[0].shape)
        pbar.update(1)
