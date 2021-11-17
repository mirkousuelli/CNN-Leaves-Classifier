import os
import numpy as np
import cv2

def transformation(imagePathDest, imagePathSrc, imageName):
	img = cv2.imread(imagePathSrc + '/' + imageName)

	# Brightness
	gamma = 4.
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
	  for i in np.arange(0, 256)]).astype("uint8")
	img = cv2.LUT(img, table)

	# Contrast
	xp = [0, 64, 128, 192, 255]
	fp = [0, 16, 128, 240, 255]
	x = np.arange(256)
	table = np.interp(x, xp, fp).astype('uint8')
	img = cv2.LUT(img, table)

	# Denoise
	img = cv2.fastNlMeansDenoisingColored(img, None, 15, 10, 50)

	# Median Blur
	img = cv2.medianBlur(img, 5)

	# GrayScale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Binary
	(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# Erosion
	kernel = np.ones((2,2), np.uint8) 
	img = cv2.erode(img, kernel, iterations=1)

	# Dilation
	kernel = np.ones((2,2), np.uint8) 
	img = cv2.dilate(img, kernel, iterations=1)

	# Edge Detection
	img = cv2.Canny(img, 100, 200)

	print(imagePathDest + imageName)
	isWritten = cv2.imwrite(imagePathDest + '/' + imageName, img)

	if isWritten:
		print('The image is successfully saved.')
  


baseDir = 'an2dl-homeworks/'
imageCV2Path = 'cv2_preprocessing/'
imagePath = 'training_validation_testing'
imagePathSub = imagePath.split('_')

for s in imagePathSub:
	src_path = baseDir + imagePath + '/mirko/' + s
	dst_path = baseDir + imageCV2Path + '/' + s
	if not os.path.exists(dst_path):
		os.makedirs(dst_path)
	
	for folder in os.listdir(src_path):
		src = src_path + '/' + str(folder)
		dst = dst_path + '/' + str(folder) 
		onlyfiles = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
		
		if not os.path.exists(dst):
			os.makedirs(dst)
		
		for imageName in onlyfiles:
			transformation(dst, src, imageName)
