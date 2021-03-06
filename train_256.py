from networks_256 import Generator, Discriminator
from ops import Hinge_loss, ortho_reg
import tensorflow as tf
import numpy as np
from utils import read_imagenet, truncated_noise_sample
from PIL import Image
import time
import scipy.io as sio
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(1)

NUMS_CLASS = 10
BETA = 1e-4
IMG_H = 256
IMG_W = 256
Z_DIM = 140
BATCH_SIZE = 8
TRAIN_ITR = 100000
TRUNCATION = 2.0

def Train():
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
    train_phase = tf.compat.v1.placeholder(tf.bool)
    z = tf.compat.v1.placeholder(tf.float32, [None, Z_DIM])
    y = tf.compat.v1.placeholder(tf.int32, [None])
    G = Generator("generator")
    D = Discriminator("discriminator")
    fake_img = G(z, train_phase, y, NUMS_CLASS)
    fake_logits = D(fake_img, y, NUMS_CLASS, None)
    real_logits = D(x, y, NUMS_CLASS, "NO_OPS")
    D_loss, G_loss = Hinge_loss(real_logits, fake_logits)
    D_ortho = BETA * ortho_reg(D.var_list())
    G_ortho = BETA * ortho_reg(G.var_list())
    D_loss += D_ortho
    G_loss += G_ortho
    D_opt = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.compat.v1.train.AdamOptimizer(4e-4, beta1=0., beta2=0.9).minimize(G_loss, var_list=G.var_list())
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    # saver.restore(sess, path_save_para+".\\model.ckpt")
    data = sio.loadmat("./dataset/imagenet_256_.mat")
    labels = data["labels"][0, :]
    data = data["data"]
    for itr in range(TRAIN_ITR):
        readtime = 0
        updatetime = 0
        for d in range(2):
            s_read = time.time()
            batch, Y = read_imagenet(data, labels, BATCH_SIZE)
            e_read = time.time()
            readtime += e_read - s_read
            batch = batch / 127.5 - 1
            Z = truncated_noise_sample(BATCH_SIZE, Z_DIM, TRUNCATION)
            s_up = time.time()
            sess.run(D_opt, feed_dict={z: Z, x: batch, train_phase: True, y: Y})
            e_up = time.time()
            updatetime += e_up - s_up

        s = time.time()
        Z = truncated_noise_sample(BATCH_SIZE, Z_DIM, TRUNCATION)
        sess.run(G_opt, feed_dict={z: Z, train_phase: True, y: Y})
        e = time.time()
        one_itr_time = e - s + updatetime + readtime
        if itr % 100 == 0:
            Z = truncated_noise_sample(BATCH_SIZE, Z_DIM, TRUNCATION)
            Dis_loss = sess.run(D_loss, feed_dict={z: Z, x: batch, train_phase: False, y: Y})
            Gen_loss = sess.run(G_loss, feed_dict={z: Z, train_phase: False, y: Y})
            print("Iteration: %d, D_loss: %f, G_loss: %f, Read_time: %f, Updata_time: %f, One_itr_time: %f" % (itr, Dis_loss, Gen_loss, readtime, updatetime, one_itr_time))
            FAKE_IMG = sess.run(fake_img, feed_dict={z: Z, train_phase: False, y: Y})
            Image.fromarray(np.uint8((FAKE_IMG[0, :, :, :] + 1)*127.5)).save("./save_img/"+str(itr) + "_" + str(Y[0]) + ".jpg")
        if itr % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")

if __name__ == "__main__":
    Train()
