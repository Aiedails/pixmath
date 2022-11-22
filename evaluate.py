# TODO: like train.py, choose some data and evaluate them
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

#===============================================================================================
#= Imports

import os
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import variable_logger
from model.network import Encoder, Decoder
from dataloader.dataloader import dataset, collate_batch, START, PAD


#===============================================================================================
#= Parameters

device = "cpu"
ckpt_path = "./checkpoints/"
batch_size = 4
max_epoch = 4
save_every = 200


input_size = (200, 60)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)



test_sets = {
    "train": {"groundtruth": "./data/groundtruth_train.tsv", "root": "./data/imgs/converted/",}
}
use_cuda = torch.cuda.is_available()


#===============================================================================================
#= Functions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        nargs="+",
        required=True,
        default=[ckpt_path],
        help="Path to the checkpoint to be used for the evaluation",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        default=batch_size,
        type=int,
        help="Size of data batches [default: {}]".format(batch_size),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        default=["train"],
        type=str,
        choices=test_sets.keys(),
        nargs="+",
        help="Dataset used for evaluation [default: {}]".format("train"),
    )
    # parser.add_argument(
    #     "-w",
    #     "--workers",
    #     dest="num_workers",
    #     default=num_workers,
    #     type=int,
    #     help="Number of workers for loading the data [default: {}]".format(num_workers),
    # )
    # parser.add_argument(
    #     "--beam-width",
    #     dest="beam_width",
    #     default=beam_width,
    #     type=int,
    #     help="Width of the beam [default: {}]".format(beam_width),
    # )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    # parser.add_argument(
    #     "--prefix",
    #     dest="prefix",
    #     default="",
    #     type=str,
    #     help="Prefix of checkpoint names",
    # )

    return parser.parse_args()

def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)


def evaluate(enc, dec, data_loader, device, checkpoint):

    for data in data_loader:

        # Copied from test.py
        img = data["img"]
        res = data["truth"]
        expected = res["encoded"].to(device)
        expected[expected == -1] = data_loader.dataset.token_to_id[PAD]

        curr_batch_size = len(img)
        batch_max_len = expected.size(1)

        hidden = dec.init_hidden(batch_size).to(device)
        sequence = torch.full(
            (curr_batch_size, 1),
            data_loader.dataset.token_to_id[START],
            dtype=torch.long,
            device=device,
        )

        enc_low_res, enc_high_res = enc(img)

        #print(enc_low_res.shape)

        # decoded_values = []
        # for i in range(batch_max_len - 1):
        #     previous = expected[:, i] # on all batch
        #     previous = previous.view(-1, 1)
        #     out, hidden = dec(previous, hidden, A, B)
        #     hidden = hidden.detach()
        #     _, top1_id = torch.topk(out, 1) # use hard max on rnn selection
        #     sequence = torch.cat((sequence, top1_id), dim=1)
        #     decoded_values.append(out)
        # # print(sequence)
        # # exit(0)
        # decoded_values = torch.stack(decoded_values, dim=2).to(device)
        # # print(decoded_values.shape)
        # # print(expected.shape)
        # loss = criterion(decoded_values, expected[:, 1:].contiguous())


        pass



    return

#===============================================================================================
#= Main

def main():
    options = parse_args()
    is_cuda = use_cuda and not options.no_cuda
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    # Can pass multiple datasets and checkpoints
    for dataset_name in options.dataset:
        results = {"best": {}, "mean": {}, "highest_prob": {}}

        for checkpoint_path in options.checkpoint:
            checkpoint_name, _ = os.path.splitext(os.path.basename(checkpoint_path))


            # Loading the checkpoint
            checkpoint = (load_checkpoint(checkpoint_path, cuda=is_cuda))

            # encoder_checkpoint = checkpoint["model"].get("encoder")
            # decoder_checkpoint = checkpoint["model"].get("decoder")

            # Loading the dataset
            data_path = test_sets[dataset_name]["root"]
            data_set = dataset(data_path, tokens_file="./data/tokens.tsv", gt_file="./data/data.txt")
            data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True)

            len_word_list = len(data_set.token_to_id)


            #### WARNING!! ####
            # Autoloading the latest checkpoint! checkpoint passed as argument not used.
            enc = Encoder().to(device)
            dec = Decoder(len_word_list, low_res_shape, high_res_shape).to(device)

            _ = enc.auto_load(ckpt_path, "encoder")
            _ = dec.auto_load(ckpt_path, "decoder")
            enc.eval()
            dec.eval()

            result = evaluate(
                enc,
                dec,
                data_loader=data_loader,
                device=device,
                checkpoint=checkpoint,
                # beam_width=options.beam_width,
                # prefix=options.prefix,
            )


    # criterion = nn.CrossEntropyLoss().to(device) # the loss function

    pass



if __name__ == "__main__":
    main()
