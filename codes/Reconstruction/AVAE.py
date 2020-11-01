from __future__ import print_function

import argparse
from random import shuffle
import numpy as np
import random
import os
import tensorflow as tf
import glob
import cv2
 
from image_reader import *
from VAE_net import *

parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--ckpt_dir", default='./snapshots', help="path of ckpt")
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs")
parser.add_argument("--image_size", type=int, default=256, help="load image size")
parser.add_argument("--feature_num", type=int, default=1, help="load feature num")
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")
parser.add_argument('--base_lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epoch')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument("--write_pred_every", type=int, default=500, help="times to write.")
parser.add_argument("--save_pred_every", type=int, default=500, help="times to save.")
parser.add_argument("--train_picture_format", default='.jpg', help="format of training datas.")
parser.add_argument("--train_picture_path", default='./picture/', help="path of training datas.")
 
args = parser.parse_args()
EPS = 1e-12

def cv_inv_proc(img):
	img_rgb = (img + 1.) * 127.5
	return img_rgb.astype(np.float32)

def get_write_picture(gen_label, label, height, width):
	gen_label_image = cv_inv_proc(gen_label[0]) 
	label_image = cv_inv_proc(label) 
	inv_gen_label_image = cv2.resize(gen_label_image, (width, height))
	inv_label_image = cv2.resize(label_image, (width, height))
	output = np.concatenate((inv_gen_label_image, inv_label_image), axis=1)
	return output

def sampler(mean, std):
	eps = tf.random_normal(
		tf.shape(std), dtype=tf.float32, mean=0, stddev=1.0, name='epsilon')
	z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(std)), eps))
	return z

def main():
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*"))
	tf.set_random_seed(args.random_seed)
	train_picture = tf.placeholder(
		tf.float32, shape=[1, args.image_size, args.image_size, 3], name='train_picture')

	z_mean, z_log_sigma_sq = encoder(image=train_picture, reuse=False, name='encoder')
	ext_features = sampler(z_mean, z_log_sigma_sq)
	gen_label = decoder(ext_features, reuse=False, name='decoder')

	dis_real = discriminator(image=train_picture, reuse=False, name="discriminator")
	dis_fake = discriminator(image=gen_label, reuse=True, name="discriminator")

	reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(gen_label, train_picture), 2.0))
	latent_loss = - tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
	vae_loss = tf.reduce_mean(reconstr_loss + latent_loss)

	dis_loss = tf.reduce_mean(-(tf.log(dis_real + EPS) + tf.log(1 - dis_fake + EPS)))

	vae_vars = [
		v for v in tf.trainable_variables() if 'encoder' in v.name or 'decoder' in v.name]
	d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

	vae_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)
	d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)

	d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars)
	d_train = d_optim.apply_gradients(d_grads_and_vars)
	vae_grads_and_vars = vae_optim.compute_gradients(vae_loss, var_list=vae_vars)
	vae_train = vae_optim.apply_gradients(vae_grads_and_vars)

	train_op = tf.group(d_train, vae_train)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	saver=tf.train.Saver(tf.global_variables())
	sess = tf.Session(config=config)

	init = tf.global_variables_initializer()
	sess.run(init)

	counter = 0
 
	for epoch in range(args.epoch):
		shuffle(train_picture_list) 
		for step in range(len(train_picture_list)):
			counter += 1
			picture_name, _ = os.path.splitext(os.path.basename(train_picture_list[step]))
			picture_resize, picture_height, picture_width = ImageReader(
				file_name=picture_name, picture_path=args.train_picture_path,
				picture_format=args.train_picture_format, size=args.image_size)
			batch_picture = np.expand_dims(
				np.array(picture_resize).astype(np.float32), axis = 0)
			feed_dict = {train_picture : batch_picture}
			vae_loss_value, d_loss_value, _ = sess.run(
				[vae_loss, dis_loss, train_op], feed_dict=feed_dict)
			if counter % args.save_pred_every == 0:
				saver.save(sess, args.ckpt_dir+'/ckpt.ckpt')
			if counter % args.write_pred_every == 0:
				gen_label_value = sess.run(gen_label, feed_dict=feed_dict)
				write_image = get_write_picture(
					gen_label_value, picture_resize, picture_height, picture_width)
				write_image_name = args.out_dir + "/out"+ str(counter) + ".png"
				cv2.imwrite(write_image_name, write_image)
				print(
					'epoch {:d} step {:d} vae loss = {:.5f} dis loss = {:.5f}'.format(
						epoch, step, vae_loss_value, d_loss_value))
main()
