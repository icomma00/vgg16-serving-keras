import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops
import os

def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images
# 此处需要特别注意：index_word.txt中最后一行必须要回车换行
def print_answer(argmax):
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
    print(synset[argmax])
    return synset[argmax]

# 批量展示图片
def sample_stack(stack, rows=6, cols=6, start_with=0, show_every=5):
    """
    批量展示图片，很好用的工具
    args:
    stack: shape:(N,H,W),  value range:[0-1]
    show_every:可以调整步长
    """
    fig, ax = plt.subplots(rows,cols,figsize=[18,18])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
        ax[int(i/cols),int(i % cols)].imshow(stack[ind],cmap='gray')
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()


# 读取文件夹内所有图片
def read_directory(directory_name):
    array_of_img = [] 
    for filename in os.listdir(r"./"+directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        # print(array_of_img)
    return array_of_img
