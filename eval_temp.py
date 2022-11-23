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


# from val_dl get one image and its ground truth text

# data = next(iter(val_dl))
# img = data["img"].to(device)
# gt_text = data["text"]
# gt_text_encoded = data["text_encoded"].to(device)

# use val_dl.dataset.id_to_token to convert the id to token
def decoded_sequence(sequence):
    return [val_dl.dataset.id_to_token[i] for i in sequence]

for idx, data in enumerate(val_dl):
    img = data["img"]
    res = data["truth"]
    # filename = data["filename"]
    expected = res["encoded"].to(device)
    expected[expected == -1] = val_dl.dataset.token_to_id[PAD]

    curr_batch_size = len(img)
    batch_max_len = expected.size(1)

    hidden = dec.init_hidden(batch_size).to(device)
    sequence = torch.full(
        (curr_batch_size, 1),
        val_dl.dataset.token_to_id[START],
        dtype=torch.long,
        device=device,
    )

    A, B = enc(img)
    # print(A.shape)
    # print(B.shape)
    decoded_values = []

    for i in range(batch_max_len - 1):
       previous = sequence[:,-1] # on all batch
       previous = previous.view(-1, 1).to(device)
       out, hidden = dec(previous, hidden, A, B)
       hidden = hidden.detach()
       _, top1_id = torch.topk(out, 1) # use hard max on rnn selection
       sequence = torch.cat((sequence, top1_id), dim=1)
       decoded_values.append(out)
    # print(sequence)
    # exit(0)
    decoded_values = torch.stack(decoded_values, dim=2).to(device)
    # print(decoded_values.shape)
    # print(expected.shape)
    loss = criterion(decoded_values, expected[:, 1:])

    # use id_to_token to convert the id to token
    final_res = decoded_sequence(sequence[0].tolist())
    if PAD in final_res:
        del final_res[final_res.index("<PAD>"):] # remove padding
    
    # print("On file", filename)
    print("predict: ", " ".join(final_res[1:-2]))
    # split the gt with space too
    print("gt:      ", " ".join(res["text"]))
