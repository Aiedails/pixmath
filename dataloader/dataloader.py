import os
import csv
import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from logger import variable_logger as logger

log = logger()

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

# There are so many symbols (mostly escape sequences) that are in the test sets but not
# in the training set.


def remove_unknown_tokens(truth):
    # Remove \mathrm and \vtop are only present in the test sets, but not in the
    # training set. They are purely for formatting anyway.
    remaining_truth = truth.replace("\\mathrm", "")
    remaining_truth = remaining_truth.replace("\\vtop", "")
    # \; \! are spaces and only present in 2014's test set
    remaining_truth = remaining_truth.replace("\\;", " ")
    remaining_truth = remaining_truth.replace("\\!", " ")
    remaining_truth = remaining_truth.replace("\\ ", " ")
    # There's one occurrence of \dots in the 2013 test set, but it wasn't present in the
    # training set. It's either \ldots or \cdots in math mode, which are essentially
    # equivalent.
    remaining_truth = remaining_truth.replace("\\dots", "\\ldots")
    # Again, \lbrack and \rbrack where not present in the training set, but they render
    # similar to \left[ and \right] respectively.
    remaining_truth = remaining_truth.replace("\\lbrack", "\\left[")
    remaining_truth = remaining_truth.replace("\\rbrack", "\\right]")
    # Same story, where \mbox = \leavemode\hbox
    remaining_truth = remaining_truth.replace("\\hbox", "\\mbox")
    # There is no reason to use \lt or \gt instead of < and > in math mode. But the
    # training set does. They are not even LaTeX control sequences but are used in
    # MathJax (to prevent code injection).
    # remaining_truth = remaining_truth.replace("<", "\\lt")
    # remaining_truth = remaining_truth.replace(">", "\\gt")
    # \parallel renders to two vertical bars
    remaining_truth = remaining_truth.replace("\\parallel", "||")
    # Some capital letters are not in the training set...
    remaining_truth = remaining_truth.replace("O", "o")
    remaining_truth = remaining_truth.replace("W", "w")
    remaining_truth = remaining_truth.replace("\\Pi", "\\pi")
    return remaining_truth


def load_vocab(tokens_file):
    with open(tokens_file, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        tokens = next(reader)
        tokens.extend(SPECIAL_TOKENS)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token

# Rather ignorant way to encode the truth, but at least it works.


def encode_truth(truth, token_to_id):
    truth_tokens = []
    remaining_truth = remove_unknown_tokens(truth).strip()
    while len(remaining_truth) > 0:
        try:
            matching_starts = [
                [i, len(tok)]
                for tok, i in token_to_id.items()
                if remaining_truth.startswith(tok)
            ]
            # Take the longest match
            index, tok_len = max(matching_starts, key=lambda match: match[1])
            truth_tokens.append(index)
            remaining_truth = remaining_truth[tok_len:].lstrip()
        except ValueError:
            # print(truth)
            # print(remaining_truth)
            raise Exception("Truth contains unknown token")
    return truth_tokens

def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "img": torch.stack([d["img"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded),
        },
    }



class dataset(Dataset):
    def __init__(self, path: str, tokens_file: str, gt_file: str, device="cpu"):
        self.path = path
        self.filenames = os.listdir(path)
        self.images = []
        self.device = device
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)

        temp_img_matches = open("./data/100_200_30_60_img_number.txt", "r").read().splitlines()
        self.temp_file_matches = temp_img_matches

        with open(gt_file, "r") as fd:
            reader = csv.reader(fd, delimiter="\t")
            self.truth = [{
                "text": truth[0],
                "encoded": [
                    self.token_to_id[START],
                    *encode_truth(truth[0], self.token_to_id),
                    self.token_to_id[END],
                ],
            } for truth in reader
            ]

        # self.transforms = ToTensor()

    def __getitem__(self, index):
        data = {}
        img = (read_image(os.path.join(self.path, self.filenames[index]),
                          torchvision.io.ImageReadMode.GRAY) / 255.0).to(self.device)
        data["img"] = img
        # data["truth"] = self.truth[index]
        data["truth"] = self.truth[int(self.temp_file_matches[index])]
        # print(self.filenames[index], int(self.temp_file_matches[index]))
        # print(self.truth[index])
        return data

    def __len__(self):
        return len(self.filenames)
