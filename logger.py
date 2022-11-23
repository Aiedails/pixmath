from termcolor import colored

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from torch.utils.tensorboard.writer import SummaryWriter

class logger():
    def __init__(self, path):
        self.writer = SummaryWriter(log_dir=path)
    def log(self, name:str, loss:float, n_iter:int):
        self.writer.add_scalar(name, loss, global_step = n_iter)
    def log_image(self, name:str, image:torch.Tensor, dataformats="CHW"):
        self.writer.add_image(name, np.array(image.detach().cpu()), dataformats=dataformats)
    def log_image_batched(self, name:str, image:torch.Tensor, img_formats="CHW"):
        # TODO: deal with the condition of [B,C,H,W]. log B images into a grid
        imgs = np.array(image.detach().cpu())
        for i in range(imgs.shape[0]):
            self.writer.add_image(name + "["+str(i)+"]", imgs[i], img_formats)
    def log_image_list(self, name:str, image:list[torch.Tensor], img_formats="CHW"):
        # TODO: deal with the condition of a list of [C,H,W]. log all images into a grid
        pass
    def log_image_batched(self, name:str, text:str, n_iter:int):
        self.writer.add_text(name, text)


class variable_logger():
    def __init__(self, is_active=True):
        self.book = {}
        self.is_active = is_active
    def log(self, name:str, value, file=None, func=None, is_once:bool=False):
        if not self.is_active:
            return
        if is_once == True and name in self.book.keys():
            return
        if name not in self.book.keys():
            self.book[name] = value
        if(file == None):
            print(colored("logger:", "blue"), colored(f"{name}", "yellow"), "\t", f"{value};")
        elif(file != None and func == None):
            print(colored("logger:", "blue"), colored(f"{name}", "yellow"), "\t", f"{value}", "\t", colored(f"from {file};", "green"))
        elif(file != None and func != None):
            print(colored("logger:", "blue"), colored(f"{name}", "yellow"), "\t", f"{value}", "\t", colored(f"from {file}", "green"), "in func", colored(f"{func};", "red"))
    def log_all(self):
        print("logger: ", end="")
        for i in self.book.keys():
            print(i, "- ", f"{self.book[i]:.5f}", sep="", end="; ")
        print("\n", end="")
    def update(self, name:str, value):
        self.book[name] = value
