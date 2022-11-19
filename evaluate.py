import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import variable_logger
from model.network import Encoder, Decoder
from dataloader.dataloader import dataset, collate_batch, START, PAD

device = "cpu"
ckpt_path = "./checkpoints/"
batch_size = 4
max_epoch = 4
save_every = 200


input_size = (200, 60)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)


# TODO: like train.py, choose some data and evaluate them

validation_path = "./data/imgs/converted/" # for debug, let it's same as that of training data

ds = dataset(validation_path, tokens_file="./data/tokens.tsv", gt_file="./data/data.txt")
val_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True)
len_word_list = len(ds.token_to_id)

enc = Encoder()
dec = Decoder(len_word_list, low_res_shape, high_res_shape)
# the auto_load function check the folder and load the newest checkpoints
# return their epoch and iteration. But now we don't need them
_ = enc.auto_load(ckpt_path, "encoder")
_ = dec.auto_load(ckpt_path, "decoder")
enc.eval()
dec.eval()

criterion = nn.CrossEntropyLoss().to(device) # the loss function

# TODO: get a minibatch of data and evaluate them
# we want to see at least:
# 1. the loss
# 2. the predicted text
# 3. the ground truth text
# 4. the image
# 5. the attention map

# some tips:
# - for the attention map, you can go to check decoder.att1.alpha and decoder.att2.alpha
# - for plot and save the images, you may want to use plt.imshow or plt.imsave 
# - for the predicted text, you can use the function ds.id_to_token to convert the id to token
# - for the ground truth text, check daloader.dataset.data["text"] to get un-embedded text
# - remember that you do not need to update the model.
