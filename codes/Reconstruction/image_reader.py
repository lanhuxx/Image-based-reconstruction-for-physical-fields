import cv2

def ImageReader(file_name, picture_path, picture_format=".jpg", size=64):
	picture_name = picture_path + file_name + picture_format
	picture = cv2.imread(picture_name, 1)
	picture_height = picture.shape[0]
	picture_width = picture.shape[1]
	picture_resize_t = cv2.resize(picture, (size, size)) 
	picture_resize = picture_resize_t / 127.5 - 1.
	return picture_resize, picture_height, picture_width
