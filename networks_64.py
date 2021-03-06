from ops import *

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase, y, nums_class):
        z_dim = int(inputs.shape[-1])
        nums_layer = 4
        remain = z_dim % nums_layer
        chunk_size = (z_dim - remain) // nums_layer
        z_split = tf.split(inputs, [chunk_size] * (nums_layer - 1) + [chunk_size + remain], axis=1)
        with tf.compat.v1.variable_scope(name_or_scope=self.name, reuse=tf.compat.v1.AUTO_REUSE):
            inputs = dense("dense", inputs, 1024*4*4)
            inputs = tf.reshape(inputs, [-1, 4, 4, 1024])
            inputs = G_Resblock("ResBlock1", inputs, 1024, train_phase, z_split[0], y, nums_class)
            inputs = G_Resblock("ResBlock2", inputs, 512, train_phase, z_split[1], y, nums_class)
            inputs = G_Resblock("ResBlock3", inputs, 256, train_phase, z_split[2], y, nums_class)

            inputs = non_local("Non-local", inputs, None, True)
            inputs = G_Resblock("ResBlock4", inputs, 128, train_phase, z_split[3], y, nums_class)
            inputs = relu(conditional_batchnorm(inputs, train_phase, "BN"))
            inputs = conv("conv", inputs, k_size=3, nums_out=3, strides=1)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, y, nums_class, update_collection=None):
        with tf.compat.v1.variable_scope(name_or_scope=self.name, reuse=tf.compat.v1.AUTO_REUSE):
            inputs = D_FirstResblock("ResBlock1", inputs, 64, update_collection, is_down=True)
            inputs = non_local("Non-local", inputs, None, True)
            inputs = D_Resblock("ResBlock2", inputs, 128, update_collection, is_down=True)
            inputs = D_Resblock("ResBlock3", inputs, 256, update_collection, is_down=True)
            inputs = D_Resblock("ResBlock4", inputs, 512, update_collection, is_down=True)
            inputs = D_Resblock("ResBlock5", inputs, 1024, update_collection, is_down=True)

            inputs = relu(inputs)
            inputs = global_sum_pooling(inputs)
            temp = Inner_product(inputs, y, nums_class, update_collection)
            inputs = dense("dense", inputs, 1, update_collection, is_sn=True)
            inputs= temp + inputs
            return inputs

    def var_list(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.name)

if __name__ == "__main__":
    x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
    z = tf.compat.v1.placeholder(tf.float32, [None, 100])
    y = tf.compat.v1.placeholder(tf.float32, [None, 100])
    train_phase = tf.compat.v1.placeholder(tf.bool)
    G = Generator("generator")
    D = Discriminator("discriminator")
    fake_img = G(z, train_phase)
    fake_logit = D(fake_img)
    aaa = 0

