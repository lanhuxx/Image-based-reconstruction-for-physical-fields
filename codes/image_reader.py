import os
import numpy as np
import tensorflow as tf
import cv2

def ImageReader(file_name, picture_path, label_path, picture_format = ".jpg", label_format = ".jpg", size=64):
	picture_name = picture_path + file_name + picture_format #得到图片名称和路径
	label_name = label_path + file_name + label_format #得到标签名称和路径
	picture = cv2.imread(picture_name, 1) #读取图片
	label = cv2.imread(label_name, 1) #读取标签
	picture_height = picture.shape[0] #得到图片的高
	picture_width = picture.shape[1] #得到图片的宽
	label_height = label.shape[0]
	label_width = label.shape[1]
	picture_resize_t = cv2.resize(picture, (size, size)) #调整图片的尺寸，改变成网络输入的大小
	picture_resize = picture_resize_t / 127.5 - 1. #归一化图片
	label_resize_t = cv2.resize(label, (size, size)) #调整标签的尺寸，改变成网络输入的大小
	label_resize = label_resize_t / 127.5 - 1. #归一化标签
	return picture_resize, label_resize, picture_height, picture_width, label_height, label_width #返回网络输入的图片，标签，还有原图片和标签的长宽
