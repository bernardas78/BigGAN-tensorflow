from networks_128 import Generator
import tensorflow as tf
import numpy as np
from PIL import Image
import os

NUMS_GEN = 64
NUMS_CLASS = 40
BATCH_SIZE = 64
Z_DIM = 128
IMG_H = 128
IMG_W = 128

def generate():
    if not os.path.exists("./generate"):
        os.mkdir("./generate")
    tf.compat.v1.disable_eager_execution()
    train_phase = tf.compat.v1.placeholder(tf.bool)
    z = tf.compat.v1.random_normal([BATCH_SIZE, Z_DIM])
    y = tf.compat.v1.placeholder(tf.int32, [None])
    G = Generator("generator")
    fake_img = G(z, train_phase, y, NUMS_CLASS)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "generator"))
    #saver.restore(sess, "./save_para64/.\\model.ckpt")
    saver.restore(sess, r"./save_para/model.ckpt")

    for nums_c in range(NUMS_CLASS):
        FAKE_IMG = sess.run(fake_img, feed_dict={train_phase: False, y: nums_c * np.ones([NUMS_GEN])})
        concat_img = np.zeros([8*IMG_H, 8*IMG_W, 3])
        c = 0
        for i in range(8):
            for j in range(8):
                concat_img[i*IMG_H:i*IMG_H+IMG_H, j*IMG_W:j*IMG_W+IMG_W] = FAKE_IMG[c]
                c += 1
        Image.fromarray(np.uint8((concat_img + 1) * 127.5)).save("./generate/"+str(nums_c)+".jpg")

if __name__ == "__main__":
    generate()
