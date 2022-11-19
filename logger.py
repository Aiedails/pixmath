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
    
    # 使用下面的函数来导入loss 如果需要acc 创建新的函数即可
    def log_train_loss(self, loss:float, n_iter:int):
        self.log('train_loss', loss, n_iter)
    def log_test_loss(self, loss:float, n_iter:int):
        self.log('test_loss', loss, n_iter)
        
    def log_image(self, name:str, image:torch.Tensor, dataformats='CHW'):
        self.writer.add_image(name, image, dataformats=dataformats)
        
    # 使用下面的函数来导入图片
    def log_formula_image(self, img):
        img_array = np.array(img)
        self.log_image('formula_image', img_array, dataformats='HWC')
    

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
