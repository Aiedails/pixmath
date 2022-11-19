import numpy as np
from logger import *
import cv2
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # 随机生成100个0到10的浮点数用于测试
    x = np.random.rand(100) * 10
    # 导入一张图片用于测试
    img =  cv2.imread("test.png")

    # 创建新的对象 tensorboard文件路径："runs/result"
    Log = logger("runs/result")
    for n_iter in range(100):
        Log.log("train_loss", x[n_iter], n_iter)
    Log.log_image("formula_image", img)
