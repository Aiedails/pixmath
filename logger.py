from termcolor import colored

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.tensorboard.writer import SummaryWriter

class logger():
    def __init__(self, path):
        self.writer = SummaryWriter(path)
    def log(self, name:str, loss:float, n_iter:int):
        self.writer.add_scalar(name, loss, global_step = n_iter)

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
