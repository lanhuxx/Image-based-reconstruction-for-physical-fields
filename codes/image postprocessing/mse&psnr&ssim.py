import tensorflow as tf
from PIL import Image
import argparse

# 把输入参数打印到屏幕 
parser = argparse.ArgumentParser(description='')

parser.add_argument("--train_path", default='E:/backups/impact2/punching press/xOy/images/801/', help="path of images.")
parser.add_argument("--label_path", default='./CGAN_801_feature=1/', help="path of label.")
parser.add_argument("--image_size", type=int, default=801, help="load image number") #网络输入的尺度

args = parser.parse_args() #用来解析命令行参数

sess = tf.Session()

for i in range(args.image_size):
	dir1 = args.train_path + str(i) + '.jpg'
	dir2 = args.label + str(i) + '.jpg'
	img1 = tf.gfile.FastGFile(dir1,'rb').read()
	img1 =  tf.image.decode_jpeg(img1)
	img2 = tf.gfile.FastGFile(dir2,'rb').read()
	img2 =  tf.image.decode_jpeg(img2)
	img1 = tf.image.convert_image_dtype(img1, tf.float32)
	img2 = tf.image.convert_image_dtype(img2, tf.float32)
	psnr = tf.image.psnr(img1, img2, max_val=1.0)
	ssim = tf.image.ssim(img1, img2, max_val=1.0)
	mse = tf.reduce_mean((img1-img2) ** 2)
	print(sess.run(psnr),'\t', sess.run(ssim), '\t', sess.run(mse))
