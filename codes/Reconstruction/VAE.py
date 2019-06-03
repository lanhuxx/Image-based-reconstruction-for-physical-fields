from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
from random import shuffle
import random
import os
import sys
import math
import tensorflow as tf
import glob
import cv2
 
from image_reader import *
from VAE_net import *
import xlwt
import xlrd
import scipy.io as scio

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 把输入参数打印到屏幕 
parser = argparse.ArgumentParser(description='')

parser.add_argument("--training_out_dir", default='./train_out_level2_400_feature=8', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--ckpt_out_dir", default='./ckpt/level2_400_features=8_samples', help="path of ckpt")
parser.add_argument("--decoded_image_dir", default='E:/backups/stamping/stamping/images/decoded_level2_samples=400_features=16/', help="path of decoded images") #VAE把训练样本还原后图像保存的路径
parser.add_argument("--features_save_dir", default='./level2_features=8_400_samples.mat', help="features save dir")
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--feature_num", type=int, default=8, help="load feature num")
parser.add_argument("--sample_szie", type=int, default=400, help="load sample size")
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0001, help='initial learning rate for adam') #学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epoch')  #训练的epoch数量
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--write_pred_every", type=int, default=50, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=50, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--train_picture_format", default='.jpg', help="format of training datas.") #网络训练输入的图片的格式(图片在CGAN中被当做条件)
parser.add_argument("--train_label_format", default='.jpg', help="format of training labels.") #网络训练输入的标签的格式(标签在CGAN中被当做真样本)
parser.add_argument("--train_picture_path", default='E:/backups/stamping/stamping/case I - wjq/images/decoded_level1_samples=400_features=1/', help="path of training datas.") #网络训练输入的图片路径
parser.add_argument("--train_label_path", default='E:/backups/stamping/stamping/case I - wjq/images/decoded_level1_samples=400_features=1/', help="path of training labels.") #网络训练输入的标签路径
 
args = parser.parse_args() #用来解析命令行参数
EPS = 1e-12 #EPS用于保证log函数里面的参数大于零

def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
	img_rgb = (img + 1.) * 127.5
	return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像

def get_write_picture(picture, gen_label, label, height, width): #get_write_picture函数得到训练过程中的可视化结果
	# picture_image = cv_inv_proc(picture) #还原输入的图像
	gen_label_image = cv_inv_proc(gen_label[0]) #还原生成的样本
	label_image = cv_inv_proc(label) #还原真实的样本(标签)
	# inv_picture_image = cv2.resize(picture_image, (width, height)) #还原图像的尺寸
	inv_gen_label_image = cv2.resize(gen_label_image, (width, height)) #还原生成的样本的尺寸
	inv_label_image = cv2.resize(label_image, (width, height)) #还原真实的样本的尺寸
	output = np.concatenate((inv_gen_label_image, inv_label_image), axis=1) #把他们拼起来
	return output

def get_write_picture_single(gen_label, height, width): #get_write_picture函数得到训练过程中的可视化结果
	# picture_image = cv_inv_proc(picture) #还原输入的图像
	gen_label_image = cv_inv_proc(gen_label[0]) #还原生成的样本
	# inv_picture_image = cv2.resize(picture_image, (width, height)) #还原图像的尺寸
	inv_gen_label_image = cv2.resize(gen_label_image, (width, height)) #还原生成的样本的尺寸
	return inv_gen_label_image

def l1_loss(src, dst): #定义l1_loss
	return tf.reduce_mean(tf.abs(src - dst))

# sampler
def sampler(mean, std):
	eps = tf.random_normal(tf.shape(std), dtype=tf.float32, mean=0, stddev=1.0, name='epsilon')
	z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(std)), eps))
	return z

def main(): #训练程序的主函数
	if not os.path.exists(args.training_out_dir):
		os.makedirs(args.training_out_dir)
	if not os.path.exists(args.ckpt_out_dir):
		os.makedirs(args.ckpt_out_dir)

	train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*")) #得到训练输入图像路径名称列表
	tf.set_random_seed(args.random_seed) #初始一下随机数
	train_picture = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='train_picture') #输入的训练图像
	train_label = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size, 3],name='train_label') #输入的与训练图像匹配的标签

	z_mean, z_log_sigma_sq = encoder(image=train_picture, reuse=False, name='generator')
	ext_features = sampler(z_mean, z_log_sigma_sq)
	gen_label = decoder(ext_features, reuse=False, name='generator')

	reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(gen_label, train_label), 2.0))
	latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
	mse = tf.reduce_mean(reconstr_loss + latent_loss)
	train_op = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1).minimize(mse)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True #设定显存不超量使用

	saver=tf.train.Saver(tf.global_variables())
	sess = tf.Session(config=config) #新建会话层
	init = tf.global_variables_initializer() #参数初始化器

	sess.run(init) #初始化所有可训练参数

	counter = 0 #counter记录训练步数
 
	for epoch in range(args.epoch): #训练epoch数	
		shuffle(train_picture_list) #每训练一个epoch，就打乱一下输入的顺序
		for step in range(len(train_picture_list)): #每个训练epoch中的训练step数
			counter += 1
			picture_name, _ = os.path.splitext(os.path.basename(train_picture_list[step])) #获取不包含路径和格式的输入图片名称
			#读取一张训练图片，一张训练标签，以及相应的高和宽
			picture_resize, label_resize, picture_height, picture_width, label_height, label_width = ImageReader(
				file_name=picture_name, picture_path=args.train_picture_path, label_path=args.train_label_path,
				picture_format = args.train_picture_format, label_format = args.train_label_format, size = args.image_size)
			batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0) #填充维度
			batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0) #填充维度
			feed_dict = {train_picture: batch_picture, train_label: batch_label} #构造feed_dict
			mse_value, _ = sess.run([mse, train_op], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
			if counter % args.save_pred_every == 0:
				dir_ = args.ckpt_out_dir + '/ckpt.ckpt'
				saver.save(sess, dir_)
			if counter % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
				gen_label_value = sess.run(gen_label, feed_dict=feed_dict) #run出生成器的输出
				write_image = get_write_picture(picture_resize, gen_label_value, label_resize, label_height, label_width) #得到训练的可视化结果
				write_image_name = args.training_out_dir + "/out"+ str(counter) + ".png" #待保存的训练可视化结果路径与名称
				cv2.imwrite(write_image_name, write_image) #保存训练的可视化结果
			print('epoch {:d} step {:d} \t mse = {:.5f}'.format(epoch, step, mse_value))
# main()

def decoder_image():
	if not os.path.exists(args.decoded_image_dir):
		os.makedirs(args.decoded_image_dir)
	train_picture_list = glob.glob(os.path.join(args.train_picture_path, "*")) #得到训练输入图像路径名称列表
	# tf.set_random_seed(args.random_seed) #初始一下随机数
	train_picture = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 3], name='train_picture') #输入的训练图像
	train_label = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 3], name='train_label') #输入的与训练图像匹配的标签

	z_mean, z_log_sigma_sq = encoder(image=train_picture, reuse=False, name='generator')
	ext_features = sampler(z_mean, z_log_sigma_sq)
	gen_label = decoder(ext_features, reuse=False, name='generator')

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		model_path = args.ckpt_out_dir + '/'
		module_file = tf.train.latest_checkpoint(model_path)
		saver.restore(sess, module_file)

		for step in range(args.sample_szie):
			tpl = args.train_picture_path + '/%d.jpg'  % int(step+1)
			picture_name, _ = os.path.splitext(os.path.basename(tpl)) #获取不包含路径和格式的输入图片名称
			# print(picture_name)
			#读取一张训练图片，一张训练标签，以及相应的高和宽
			picture_resize, label_resize, picture_height, picture_width, label_height, label_width = ImageReader(
				file_name=picture_name, picture_path=args.train_picture_path, label_path=args.train_label_path,
				picture_format = args.train_picture_format, label_format = args.train_label_format, size = args.image_size)
			batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0) #填充维度
			batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0) #填充维度
			feed_dict = {train_picture: batch_picture, train_label: batch_label} #构造feed_dict
			gen_label_value = sess.run(gen_label, feed_dict=feed_dict) #run出生成器的输出
			write_image = get_write_picture_single(gen_label_value, label_height, label_width)
			write_image_name = args.decoded_image_dir+ str(int(step+1)) + ".jpg" #待保存的训练可视化结果路径与名称
			cv2.imwrite(write_image_name, write_image) #保存训练的可视化结果
			print(step)
# decoder_image()

def encoder_feature_extract():
	train_picture = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 3], name='train_picture') #输入的训练图像
	train_label = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 3], name='train_label') #输入的与训练图像匹配的标签

	z_mean, z_log_sigma_sq = encoder(image=train_picture, reuse=False, name='generator')
	ext_features = sampler(z_mean, z_log_sigma_sq)
	gen_label = decoder(ext_features, reuse=False, name='generator')

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		model_path = args.ckpt_out_dir + '/'
		module_file = tf.train.latest_checkpoint(model_path)
		saver.restore(sess, module_file)

		FS = []
		a = 0
		for step in range(args.sample_szie):
			tpl = args.train_picture_path + '/%d.jpg'  % int(step+1)
			picture_name, _ = os.path.splitext(os.path.basename(tpl)) #获取不包含路径和格式的输入图片名称
			# print(picture_name)
			#读取一张训练图片，一张训练标签，以及相应的高和宽
			picture_resize, label_resize, picture_height, picture_width, label_height, label_width = ImageReader(
				file_name=picture_name, picture_path=args.train_picture_path, label_path=args.train_label_path,
				picture_format = args.train_picture_format, label_format = args.train_label_format, size = args.image_size)
			batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0) #填充维度
			batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0) #填充维度
			feed_dict = {train_picture: batch_picture, train_label: batch_label} #构造feed_dict
			ext_features_value = sess.run(ext_features, feed_dict=feed_dict) #run出生成器的输出
			# print(ext_features_value)
			FS.append(ext_features_value[0])
			a += 1
			print(a)
		# print(FS)
		# FS = sess.run(tf.reshape(FS, [-1, 16]))
		# print(np.array(FS))
		save_fn = args.features_save_dir
		scio.savemat(save_fn, {'VAE': FS})
encoder_feature_extract()

def decoder_new_image():
	# dataFile1 = 'E:/backups/stamping/stamping/new_50000'
	# data1 = scio.loadmat(dataFile1)
	# x = data1['num']

	ext_features = tf.placeholder(tf.float32, shape=[1, args.feature_num])

	gen_label = decoder(ext_features, reuse=False, name='generator')

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		model_path = args.ckpt_out_dir + '/'
		module_file = tf.train.latest_checkpoint(model_path)
		saver.restore(sess, module_file)

		a = 0
		for i in range(50000):
			xs = [[random.uniform(-5.0072584, 1.5215675)]]
			gen_label_value = sess.run(gen_label, feed_dict={ext_features: xs}) #run出生成器的输出
			write_image = get_write_picture_single(gen_label_value, 400, 600)
			write_image_name = 'E:/backups/stamping/stamping/images/new_level1_50000/'+ str(i) + ".jpg" #待保存的训练可视化结果路径与名称
			cv2.imwrite(write_image_name, write_image) #保存训练的可视化结果
			print(i)
# decoder_new_image()
