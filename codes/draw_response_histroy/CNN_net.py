import tensorflow as tf

HIDDEN_NODES = 128

def switch_norm(x, scope='switch_norm'):
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2], keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        mean_weight = tf.nn.softmax(tf.get_variable(
            "mean_weight", [3], initializer=tf.constant_initializer(1.0)))
        var_wegiht = tf.nn.softmax(
            tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + eps))
        x = x * gamma + beta

        return x

def mish(x):
    mish_ = x * tf.nn.tanh(tf.nn.softplus(x))
    return mish_

def resBlock_conv(input, filters, kernel_size=(3, 3), act_fn=mish,
    strides=(1, 1), padding='SAME', name='resBlock_deconv_2d'):
    with tf.variable_scope(name):
        d1 = switch_norm(input, scope='d1_switch_norm_1')
        d1 = act_fn(d1)
        d1 = tf.layers.conv2d(
            d1, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding)
        d1 = switch_norm(d1, scope='d1_switch_norm_2')
        d1 = act_fn(d1)
        d1 = tf.layers.conv2d(
            d1, filters=filters, kernel_size=kernel_size,
            strides=(1, 1), padding=padding)
        
        d2 = tf.layers.conv2d(
            input, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding)
        return d1 + d2

def CNN(input_x, latent_dim=64, act_fn=mish, reuse=False, name="CNN"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        e1 = resBlock_conv(
            input=input_x, filters=latent_dim, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e1')
        e2 = resBlock_conv(
            input=e1, filters=latent_dim*2, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e2')
        e3 = resBlock_conv(
            input=e2, filters=latent_dim*4, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e3')
        e4 = resBlock_conv(
            input=e3, filters=latent_dim*8, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e4')
        e5 = resBlock_conv(
            input=e4, filters=latent_dim*8, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e5')
        e6 = resBlock_conv(
            input=e5, filters=latent_dim*8, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e6')
        e7 = resBlock_conv(
            input=e6, filters=latent_dim*8, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e7')
        e8 = resBlock_conv(
            input=e7, filters=latent_dim*8, kernel_size=(3, 3),
            act_fn=act_fn, strides=(2, 2), padding='SAME', name='e8')

        shape_before_flatten = tuple(e7.get_shape().as_list())

        e8 = tf.layers.flatten(e8)
        e9 = mish(tf.layers.dense(e8, HIDDEN_NODES, name='e9'))
        e10 = tf.layers.dense(e9, 1, name='e10')
        return e10
