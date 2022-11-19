import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from logger import variable_logger
from model.network import Encoder, Decoder
from dataloader.dataloader import dataset, collate_batch, START, PAD

device = "cpu"
path = "./data/imgs/converted/"
ckpt_path = "./checkpoints/"
batch_size = 4
max_epoch = 4
save_every = 200

ds = dataset(path, tokens_file="./data/tokens.tsv", gt_file="./data/data.txt")
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True)
len_word_list = len(ds.token_to_id)

input_size = (200, 60)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)


enc = Encoder()
dec = Decoder(len_word_list, low_res_shape, high_res_shape)
start_epoch, start_iter = enc.auto_load(ckpt_path, "encoder")
_, _ = dec.auto_load(ckpt_path, "decoder")

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


for epoch in range(start_epoch, max_epoch):
    with tqdm(
        total=len(train_dl.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for idx, data in enumerate(train_dl):
            img = data["img"]
            res = data["truth"]
            expected = res["encoded"].to(device)
            expected[expected == -1] = train_dl.dataset.token_to_id[PAD]

            curr_batch_size = len(img)
            batch_max_len = expected.size(1)

            hidden = dec.init_hidden(batch_size).to(device)
            sequence = torch.full(
                (curr_batch_size, 1),
                train_dl.dataset.token_to_id[START],
                dtype=torch.long,
                device=device,
            )

            A, B = enc(img)
            # print(A.shape)
            # print(B.shape)
            decoded_values = []
            for i in range(batch_max_len - 1):
               previous = expected[:, i] # on all batch
               previous = previous.view(-1, 1)
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
            loss = criterion(decoded_values, expected[:, 1:].contiguous())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(curr_batch_size)
            pbar.set_description(f"loss: {loss:.6f}")

            if (idx % save_every == 0):
                enc.auto_save(os.path.join(ckpt_path,"encoder"), epoch, idx)
                dec.auto_save(os.path.join(ckpt_path,"decoder"), epoch, idx)
                print(f"checkpoints saved for epoch: {epoch} iter: {idx}")
        scheduler.step()
