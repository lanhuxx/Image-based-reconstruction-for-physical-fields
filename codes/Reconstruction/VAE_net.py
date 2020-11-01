import tensorflow as tf

feature_num = 1
def make_var(name, shape, trainable=True):
	return tf.get_variable(name, shape, trainable = trainable)

def conv2d(input_, output_dim, kernel_size, stride,
	padding="SAME", name="conv2d", biased=False):
	input_dim = input_.get_shape()[-1]
	with tf.variable_scope(name):
		kernel = make_var(name = 'weights',
			shape=[kernel_size, kernel_size, input_dim, output_dim])
		output = tf.nn.conv2d(input_, kernel,
			[1, stride, stride, 1], padding=padding)
		if biased:
			biases = make_var(name='biases', shape=[output_dim])
			output = tf.nn.bias_add(output, biases)
		return output

def deconv2d(input_, output_dim, kernel_size,
	stride, padding="SAME", name="deconv2d"):
	input_dim = input_.get_shape()[-1]
	input_height = int(input_.get_shape()[1])
	input_width = int(input_.get_shape()[2])
	with tf.variable_scope(name):
		kernel = make_var(name='weights',
			shape=[kernel_size, kernel_size, output_dim, input_dim])
		output = tf.nn.conv2d_transpose(input_, kernel, 
			[1, input_height*2, input_width*2, output_dim], [1, 2, 2, 1],
			padding="SAME")
		return output

def batch_norm(input_, name="batch_norm"):
	with tf.variable_scope(name):
		input_dim = input_.get_shape()[-1]
		scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(variance + epsilon)
		normalized = (input_-mean)*inv
		output = scale*normalized + offset
		return output

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def encoder(image, gf_dim=32, reuse=False, name="encoder"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		e1 = batch_norm(
			conv2d(
				input_=image, output_dim=gf_dim, kernel_size=4, stride=2, name='e1_conv'), name='bn_e1')
		e2 = batch_norm(
			conv2d(
				input_=lrelu(e1), output_dim=gf_dim*2, kernel_size=4, stride=2, name='e2_conv'), name='bn_e2')
		e3 = batch_norm(
			conv2d(
				input_=lrelu(e2), output_dim=gf_dim*4, kernel_size=4, stride=2, name='e3_conv'), name='bn_e3')
		e4 = batch_norm(
			conv2d(
				input_=lrelu(e3), output_dim=gf_dim*8, kernel_size=4, stride=2, name='e4_conv'), name='bn_e4')
		e5 = batch_norm(
			conv2d(
				input_=lrelu(e4), output_dim=gf_dim*16, kernel_size=4, stride=2, name='e5_conv'), name='bn_e5')
		e6 = batch_norm(
			conv2d(
				input_=lrelu(e5), output_dim=gf_dim, kernel_size=4, stride=2, name='e6_conv'), name='bn_e6')
		e6_shape = e6.get_shape().as_list()
		nodes = e6_shape[1] * e6_shape[2] * e6_shape[3]
		e7 = tf.reshape(e6, [e6_shape[0], nodes])

		mean_w = tf.get_variable('weights1', [nodes, feature_num])
		mean_b = tf.get_variable(
			'biases1', [feature_num], initializer=tf.constant_initializer(0.1))
		sigma_sq_w = tf.get_variable('weights2', [nodes, feature_num])
		sigma_sq_b = tf.get_variable(
			'biases2', [feature_num], initializer=tf.constant_initializer(0.1))

		mean = tf.matmul(e7, mean_w) + mean_b
		sigma_sq = tf.matmul(e7, sigma_sq_w) + sigma_sq_b
		return mean, sigma_sq

def decoder(e, gf_dim=32, reuse=False, name="decoder"):
	dropout_rate = 0.5
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		w = tf.get_variable('weights3', [feature_num, 512])
		b = tf.get_variable('biases3', [512], initializer=tf.constant_initializer(0.1))
		e = tf.nn.relu(tf.matmul(e, w)+b)
		e = tf.reshape(e, [1, 4, 4, gf_dim])

		d1 = deconv2d(
			input_=tf.nn.relu(e), output_dim=gf_dim*16, kernel_size=4, stride=2, name='d_d3')
		d1 = tf.nn.dropout(d1, dropout_rate)
		d2 = deconv2d(
			input_=tf.nn.relu(d1), output_dim=gf_dim*8, kernel_size=4, stride=2, name='d_d4')
		d2 = tf.nn.dropout(d2, dropout_rate)
		d3 = deconv2d(
			input_=tf.nn.relu(d2), output_dim=gf_dim*4, kernel_size=4, stride=2, name='d_d5')
		d3 = tf.nn.dropout(d3, dropout_rate)
		d4 = deconv2d(
			input_=tf.nn.relu(d3), output_dim=gf_dim*2, kernel_size=4, stride=2, name='d_d6')
		d4 = tf.nn.dropout(d4, dropout_rate)
		d5 = deconv2d(
			input_=tf.nn.relu(d4), output_dim=gf_dim, kernel_size=4, stride=2, name='d_d7')
		d5 = tf.nn.dropout(d5, dropout_rate)
		d6 = deconv2d(
			input_=tf.nn.relu(d5), output_dim=3, kernel_size=4, stride=2, name='d_d8')
		return tf.nn.tanh(d6)

def discriminator(image, df_dim=32, reuse=False, name="discriminator"):
	dropout_rate = 0.5
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		h0 = lrelu(
			conv2d(
				input_=image, output_dim=df_dim, kernel_size=4, stride=2, name='d_h0_conv'))
		h1 = lrelu(
			batch_norm(
				conv2d(
					input_=h0, output_dim=df_dim*2, kernel_size=4, stride=2, name='d_h1_conv'), name='d_bn1'))
		h2 = lrelu(
			batch_norm(
				conv2d(
					input_=h1, output_dim=df_dim*4, kernel_size=4, stride=2, name='d_h2_conv'), name='d_bn2'))
		h3 = lrelu(
			batch_norm(
				conv2d(
					input_=h2, output_dim=df_dim*8, kernel_size=4, stride=2, name='d_h3_conv'), name='d_bn3'))
		h4 = lrelu(
			batch_norm(
				conv2d(
					input_=h3, output_dim=df_dim*16, kernel_size=4, stride=2, name='d_h4_conv'), name='d_bn4'))
		h5 = lrelu(
			batch_norm(
				conv2d(
					input_=h4, output_dim=df_dim*16, kernel_size=4, stride=2, name='d_h5_conv'), name='d_bn5'))
		h6 = lrelu(
			batch_norm(
				conv2d(
					input_=h5, output_dim=df_dim*16, kernel_size=4, stride=2, name='d_h6_conv'), name='d_bn6'))
		h7 = conv2d(input_=h6, output_dim=1, kernel_size=4, stride=2, name='d_h7_conv')
		h7 = tf.nn.dropout(h7, dropout_rate)
		return tf.sigmoid(h7[:][0][0][0])
