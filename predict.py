import numpy as np
import utils
import cv2
from keras import backend as K
from model.VGG16 import VGG16

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = VGG16(2)
    model.load_weights("./logs/ep006-loss0.057-val_loss0.203.h5")
    img = cv2.imread("E:/PycharmProject/datasets/cat-and-dog/train/cat.10797.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    img = np.expand_dims(img,axis = 0)
    img = utils.resize_image(img,(224,224))
    # utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img))))
    # 输出：
    # 猫