import tensorflow_addons as tfa
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array

import numpy as np

model_filename = './to_anime/anime_model.h5'

cust = {'InstanceNormalization': tfa.layers.InstanceNormalization}
model = load_model(model_filename, cust, compile=False)

def predict(image):
	print("TO ANIME")
	image_array = np.array([img_to_array(image)])
	print("shape=", image_array.shape)
	return 255.0*(model.predict(image_array)[0]+1.0)/2.0