import os
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from logger import variable_logger, logger
from model.network import Encoder, Decoder
from dataloader.dataloader import dataset, collate_batch, START, PAD

writer = logger("runs/test")

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" 
path = "./data/imgs/converted/" 
ckpt_path = "./checkpoints/"
batch_size = 4
max_epoch = 20
save_every = 200
teacher_forcing_ratio = 0.8
start_epoch, start_iter = 0, 0
ds = dataset(path, tokens_file="./data/tokens.tsv", gt_file="./data/data.txt") 
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True)
len_word_list = len(ds.token_to_id)


grad_norms = []
max_grad_norm = 5.0

input_size = (400, 60)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)


enc = Encoder().to(device)
dec = Decoder(len_word_list, low_res_shape, high_res_shape, device=device).to(device)
# start_epoch, start_iter = enc.auto_load(ckpt_path, "encoder")
# _, _ = dec.auto_load(ckpt_path, "decoder")

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)


for epoch in range(start_epoch, max_epoch):
    with tqdm(
        total=len(train_dl.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for idx, data in enumerate(train_dl):
            img = data["img"].to(device)
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
            decoded_values = []
            # The teacher forcing is done per batch, not symbol
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            for i in range(batch_max_len - 1):
                previous = expected[:, i] if use_teacher_forcing else sequence[:,-1]# on all batch
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
            loss = criterion(decoded_values, expected[:, 1:].contiguous())
            writer.log("loss", loss.item(), epoch * len(train_dl) + idx)

            optim_params = [
                p
                for param_group in optimizer.param_groups
                for p in param_group["params"]
            ]
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients, it returns the total norm of all parameters
            grad_norm = nn.utils.clip_grad_norm_(
                optim_params, max_norm=max_grad_norm
            )
            grad_norms.append(grad_norm)
            optimizer.step()

            correct_symbols = torch.sum(sequence == expected, dim=(0, 1)).item()
            total_symbols = expected.numel()

            pbar.update(curr_batch_size)
            pbar.set_description(f"loss: {loss:.6f}, acc: {correct_symbols / total_symbols:.2f}")

            if (idx % save_every == 0):
                enc.auto_save(os.path.join(ckpt_path,"encoder"), epoch, idx)
                dec.auto_save(os.path.join(ckpt_path,"decoder"), epoch, idx)
                print(f"checkpoints saved for epoch: {epoch} iter: {idx}")
        scheduler.step()
