import tensorflow as tf
from PIL import Image

sess = tf.Session()

for i in range(801):
	dir1 = 'E:/backups/impact2/punching press/xOy/images/801/%d.jpg' % int(i+1)
	dir2 = './CGAN_801_feature=1/%d.jpg' % int(i+1)
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
