from ops import *

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase, y, nums_class):
        z_dim = int(inputs.shape[-1])   # = 128
        nums_layer = 6
        remain = z_dim % nums_layer     # = 128 % 6 = 2
        chunk_size = (z_dim - remain) // nums_layer # = (128-2)//6 = 21
        z_split = tf.split(inputs, [chunk_size] * (nums_layer - 1) + [chunk_size + remain], axis=1)  #  [21 21 21 21 21 23]
        with tf.compat.v1.variable_scope(name_or_scope=self.name, reuse=tf.compat.v1.AUTO_REUSE):
            inputs = dense("dense", inputs, 1024*4*4)
            inputs = tf.reshape(inputs, [-1, 4, 4, 1024])
            inputs = G_Resblock("ResBlock1", inputs, 1024, train_phase, z_split[0], y, nums_class)
            print ("XXX.1 inputs.shape: {}".format(inputs.shape))
            inputs = G_Resblock("ResBlock2", inputs, 512, train_phase, z_split[1], y, nums_class)
            print ("XXX.2 inputs.shape: {}".format(inputs.shape))
            inputs = G_Resblock("ResBlock2.5", inputs, 512, train_phase, z_split[2], y, nums_class)
            print ("XXX.2.5 inputs.shape: {}".format(inputs.shape))
            inputs = G_Resblock("ResBlock3", inputs, 256, train_phase, z_split[3], y, nums_class)
            print ("XXX.3 inputs.shape: {}".format(inputs.shape))
            inputs = non_local("Non-local", inputs, None, True)     # moved non_local here due to memory constrains
            inputs = G_Resblock("ResBlock4", inputs, 128, train_phase, z_split[4], y, nums_class)
            print ("XXX.4 inputs.shape: {}".format(inputs.shape))
            #inputs = non_local("Non-local", inputs, None, True)
            print ("XXX.5 inputs.shape: {}".format(inputs.shape))
            inputs = G_Resblock("ResBlock5", inputs, 64, train_phase, z_split[5], y, nums_class)
            print ("XXX.6 inputs.shape: {}".format(inputs.shape))
            inputs = relu(conditional_batchnorm(inputs, train_phase, "BN"))
            print("XXX.7 inputs.shape: {}".format(inputs.shape))
            inputs = conv("conv", inputs, k_size=3, nums_out=3, strides=1)
            print("XXX.8 inputs.shape: {}".format(inputs.shape))
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, y, nums_class, update_collection=None):
        with tf.compat.v1.variable_scope(name_or_scope=self.name, reuse=tf.compat.v1.AUTO_REUSE):
            print("DDD.0 inputs.shape: {}".format(inputs.shape))
            inputs = D_FirstResblock("ResBlock1", inputs, 64, update_collection, is_down=True)
            print("DDD.1 inputs.shape: {}".format(inputs.shape))
            inputs = D_Resblock("ResBlock2", inputs, 128, update_collection, is_down=True)
            print("DDD.2 inputs.shape: {}".format(inputs.shape))

            inputs = non_local("Non-local", inputs, None, True)
            inputs = D_Resblock("ResBlock3.-1", inputs, 256, update_collection, is_down=True)
            print("DDD.3.-1 inputs.shape: {}".format(inputs.shape))
            inputs = D_Resblock("ResBlock3", inputs, 256, update_collection, is_down=True)
            print("DDD.3 inputs.shape: {}".format(inputs.shape))
            inputs = D_Resblock("ResBlock4", inputs, 512, update_collection, is_down=True)
            print("DDD.4 inputs.shape: {}".format(inputs.shape))
            inputs = D_Resblock("ResBlock5", inputs, 1024, update_collection, is_down=True)
            print("DDD.5 inputs.shape: {}".format(inputs.shape))
            inputs = D_Resblock("ResBlock6", inputs, 1024, update_collection, is_down=False)
            print("DDD.6 inputs.shape: {}".format(inputs.shape))
            inputs = relu(inputs)
            inputs = global_sum_pooling(inputs)
            temp = Inner_product(inputs, y, nums_class, update_collection)
            inputs = dense("dense", inputs, 1, update_collection, is_sn=True)
            inputs = temp + inputs
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

