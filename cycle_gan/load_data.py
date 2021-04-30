from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import pickle

import numpy as np


import os

def dataset_to_array(path, n=None):
	images_array = []
	image_names = os.listdir(path)[:n]
	for image_name in image_names:
		image = load_img(path+image_name)
		image_array = img_to_array(image)
		images_array.append(image_array)
	return np.array(images_array)

def save(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

# save(dataset_to_array("./data/trainA/"), "./data/trainA/trainA.pkl")
# save(dataset_to_array("./data/trainB/"), "./data/trainB/trainB.pkl")
# USAGE
# print(dataset_to_array("./data/trainA/"))