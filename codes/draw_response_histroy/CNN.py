from __future__ import print_function

import argparse
from random import shuffle
import random
import os
import tensorflow as tf
import glob
import cv2
import time
 
from image_reader import *
from VAE_net import *
from CNN_net import *
import scipy.io as scio

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(description='')

parser.add_argument("--sample_size", type=int, default=563, help="load sample size")
parser.add_argument("--image_size", type=int, default=256, help="load image size")
parser.add_argument("--output_size", type=int, default=1, help="load output size")
parser.add_argument("--batch_size", type=int, default=20, help="load batch size")
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")
parser.add_argument("--feature_num", type=int, default=3, help="load feature num")
parser.add_argument('--base_lr', type=float, default=0.000001, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=100000, help='# of epoch')
parser.add_argument("--train_picture_path",
	default='C:/2.backup/1.impact/training_data_acce/total_impact_seq_200_1440/',
	help="path of training datas.")

args = parser.parse_args()

dataFile = 'C:/2.backup/1.impact/training_data_acce/log_OFs_seq_200_1440'
data = scio.loadmat(dataFile)
LABELS = data['OF']

def sampler(mean, std):
	eps = tf.random_normal(
		tf.shape(std), dtype=tf.float32, mean=0, stddev=1.0, name='epsilon')
	z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(std)), eps))
	return z

def main(LOAD_MODEL=True):
	time0 = time.time()

	train_picture = tf.placeholder(tf.float32,
		shape=[args.batch_size, args.image_size, args.image_size, 3],
		name='train_picture')
	train_label = tf.placeholder(tf.float32,
		shape=[args.batch_size, args.output_size],
		name='train_label')

	pred_label = CNN(gen_label, name="CNN")

	loss = tf.reduce_mean((train_label - pred_label) ** 2)

	# train_op = tf.train.AdamOptimizer(LR).minimize(loss)
	optimizer = tf.train.AdamOptimizer(args.base_lr)
	gradients = optimizer.compute_gradients(loss)
	capped_gradients = [
		(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
	train_op = optimizer.apply_gradients(capped_gradients)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	saver=tf.train.Saver(tf.global_variables())
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()

	sess.run(init)

	if LOAD_MODEL: 
		print(" [*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(
			'./ckpt')

		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			saver.restore(sess, os.path.join(
				'./ckpt',
				ckpt_name))
			print('Loading success')

	R21 = 0

	# filename = 'C:/2.backup/1.impact/1.time-based optimization/2.CNN_OFs/CNN_LOSS.txt'
	# with open(filename, 'w') as file_object:
	for epoch in range(args.epoch):
		arr = np.arange(args.sample_size)
		np.random.shuffle(arr)

		Y = []
		P = []
		for step in range(int(args.sample_size/args.batch_size/20)):
			start = (step * args.batch_size) % args.sample_size
			end = min(start + args.batch_size, args.sample_size)

			batch_seq = arr[start: end]
			batch_data = []
			batch_label = []
			for i in range(len(batch_seq)):
				datai, h, w = ImageReader(
					file_name=str(batch_seq[i]), picture_path=args.train_picture_path,
					picture_format='.jpg', size=args.image_size)
				batch_data.append(datai)
				batch_label.append(LABELS[int(batch_seq[i])])
			feed_dict = {train_picture: batch_data, train_label: batch_label}

			PRED, YY, _ = sess.run([pred_label, train_label, train_op], feed_dict=feed_dict)
			Y.append(tf.convert_to_tensor(YY))
			P.append(tf.convert_to_tensor(PRED))
		
		Y = tf.reshape(Y, [-1])
		P = tf.reshape(P, [-1])

		acc = sess.run(tf.reduce_mean(1 - tf.abs((P - Y) / Y)))
		R2 = sess.run(1 - tf.reduce_mean(
			tf.reduce_mean((Y - P) ** 2) / tf.reduce_mean((Y - tf.reduce_mean(Y)) ** 2)))
		print(epoch, acc, R2)
		if R2 > R21:
			model_dir = './ckpt/ckpt.ckpt'
			saver.save(sess, model_dir)
			R21 = R2
main()
