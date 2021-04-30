import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Activation, LeakyReLU
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.preprocessing.image import array_to_img

from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec


import numpy as np
from numpy.random import randint

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

from load_data import dataset_to_array

# Save Configuration
path_save_root = "./"
path_dataset = path_save_root+"data/"
path_results_while_train = path_save_root+"results/while_train/"
path_models = path_save_root+"models/"
filename_prefix_model = 'anime_'

# Train Configuration
batch_size = 1
num_epochs = 100


# Residual Network
def get_residual_network(num_kernels, old_model):
	random_weights = RandomNormal(stddev=0.02) # random weight initialization
	# Layer 1
	resnet = Conv2D(num_kernels, (3,3), padding='same', kernel_initializer=random_weights)(old_model)
	resnet = tfa.layers.InstanceNormalization(axis=-1)(resnet)
	resnet = Activation('relu')(resnet)
	# Layer 2
	resnet = Conv2D(num_kernels, (3,3), padding='same', kernel_initializer=random_weights)(resnet)
	resnet = tfa.layers.InstanceNormalization(axis=-1)(resnet)
	resnet = Concatenate()([resnet, old_model]) # Merge feature maps of old model and resnet
	return resnet
 
# define the standalone generator model
def get_generator(input_dimensions, num_resnet_blocks=9):
	random_weights = RandomNormal(stddev=0.02) # random weight initialization
	src_image = Input(shape=input_dimensions) # 256x256x3
	
	# Layer 1
	model = Conv2D(64, (7,7), padding='same', kernel_initializer=random_weights)(src_image) # 256x256x64
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = Activation('relu')(model)
	
	# Layer 2
	model = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 128x128x128
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = Activation('relu')(model)
	
	# Layer 3
	model = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 64x64x256
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = Activation('relu')(model)
	
	# Layer 4-21 (9 resnets with 2 Conv layers each)
	for _ in range(num_resnet_blocks):
		model = get_residual_network(256, model) # 64x64x2560
	
	# Layer 22
	model = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 128x128x128
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = Activation('relu')(model)
	
	# Layer 23
	model = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 256x256x64
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = Activation('relu')(model)
	
	# Layer 24
	model = Conv2D(3, (7,7), padding='same', kernel_initializer=random_weights)(model) # 256x256x3
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	output = Activation('tanh')(model)

	model = Model(src_image, output)
	return model

class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

# Discriminator Model
def get_discriminator(input_dimensions=(256, 256, 3)):
	random_weights = RandomNormal(stddev=0.02) # random weight initialization
	src_image = Input(shape=input_dimensions) # 256x256x3

	# Hidden Layer 1
	model = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=random_weights)(src_image) # 128x128x64
	model = LeakyReLU(alpha=0.2)(model)
	model = Attention(64)(model)
	# Hidden Layer 2
	model = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 64x64x128
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = LeakyReLU(alpha=0.2)(model)

	# Hidden Layer 3
	model = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 32x32x256
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = LeakyReLU(alpha=0.2)(model)

	# Hidden Layer 4
	model = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=random_weights)(model) # 16x16x512
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = LeakyReLU(alpha=0.2)(model)

	# Hidden Layer 5
	model = Conv2D(512, (4,4), padding='same', kernel_initializer=random_weights)(model) # 16x16x512
	model = tfa.layers.InstanceNormalization(axis=-1)(model)
	model = LeakyReLU(alpha=0.2)(model)
	
	# Output Layer
	output = Conv2D(1, (4,4), padding='same', kernel_initializer=random_weights)(model) # 16x16x1

	model = Model(src_image, output)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model
 
# Cycle GAN
def make_cycle(generator_1, d_model, generator_2, input_dimensions):
	generator_1.trainable = True
	d_model.trainable = False
	generator_2.trainable = False

	# adversarial component
	input_gen = Input(shape=input_dimensions)
	gen1_out = generator_1(input_gen)
	output_d = d_model(gen1_out)

	# identity mapping component
	input_id = Input(shape=input_dimensions)
	output_id = generator_1(input_id)

	# forward cycle component
	output_f = generator_2(gen1_out)

	# backward cycle component
	gen2_out = generator_2(input_id)
	output_b = generator_1(gen2_out)

	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b]) # make model
	
	adam_optimizer = Adam(lr=0.0002, beta_1=0.5) # adam optimizer
	# compile model
	# Identity Loss (MAE) 5 times more important than Adversarial
	# Forward and backward cycle losses (MAE) are 10 times more important than Adversarial
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=adam_optimizer)
	return model


# sample images from dataset
def get_real_images(dataset, n, dim):
	random_xs = randint(0, dataset.shape[0], n)
	X = dataset[random_xs]
	y = np.ones((n, dim, dim, 1))
	return X, y

# generate fake images
def get_fake_images(model, dataset, dim):
	X = model.predict(dataset)
	y = np.zeros((len(X), dim, dim, 1))
	return X, y

def refresh_pool(pool, images, pool_size=50):
	picked = []
	for image in images:
		if len(pool) < pool_size: # add to the pool if the pool has space
			pool.append(image)
			picked.append(image)
		elif np.random.random() < 0.5: # pick this image and dont add it to pool
			picked.append(image) 
		else: # pick a random image from the pool and add this image to the pool
			ix = randint(0, len(pool))
			picked.append(pool[ix])
			pool[ix] = image
	return np.asarray(picked)


# A=human face, B=anime face
input_dimensions = (256,256,3)

generator_A2B = get_generator(input_dimensions)
generator_B2A = get_generator(input_dimensions)

discriminator_A = get_discriminator(input_dimensions)
discriminator_B = get_discriminator(input_dimensions)

cycle_A2B = make_cycle(generator_A2B, discriminator_B, generator_B2A, input_dimensions)
cycle_B2A = make_cycle(generator_B2A, discriminator_A, generator_A2B, input_dimensions)

def train(discriminator_A, discriminator_B, generator_A2B, generator_B2A, cycle_A2B, cycle_B2A, dataset):
	patch_size = discriminator_A.output_shape[1] # patch size for discriminator
	trainA, trainB = dataset
	poolA, poolB = [], [] # image pools
	num_train_steps = len(trainA) * num_epochs
	# manually enumerate epochs
	for i in range(num_train_steps):
		# get real batch
		X_realA, y_realA = get_real_images(trainA, batch_size, patch_size)
		X_realB, y_realB = get_real_images(trainB, batch_size, patch_size)
		
		# get fake batch
		X_fakeA, y_fakeA = get_fake_images(generator_B2A, X_realB, patch_size)
		X_fakeB, y_fakeB = get_fake_images(generator_A2B, X_realA, patch_size)
		
		# refresh pools
		X_fakeA = refresh_pool(poolA, X_fakeA)
		X_fakeB = refresh_pool(poolB, X_fakeB)
		
		# train generator B using adversarial and cycle loss
		g_loss2, _, _, _, _  = cycle_B2A.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

		# train discriminator A
		dA_loss1 = discriminator_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = discriminator_A.train_on_batch(X_fakeA, y_fakeA)
		
		# train generator A using adversarial and cycle loss
		g_loss1, _, _, _, _ = cycle_A2B.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		
		# train discriminator for B
		dB_loss1 = discriminator_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = discriminator_B.train_on_batch(X_fakeB, y_fakeB)
		
		print('>%d, dA=(%.3f,%.3f) dB(%.3f,%.3f) g(%.3f,%.3f])' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
		if not (i+1)%2:
			y = generator_A2B.predict([X_realA])[0]
			y_image = array_to_img(y)
			y_image.save(path_results_while_train + "%d.png" % (i+1))
			filename = path_models+filename_prefix_model+'%03d.h5' % (i+1)
			generator_A2B.save(filename)

# load dataset
trainA = dataset_to_array(path_dataset+"trainA/", 1)
trainB = dataset_to_array(path_dataset+"trainB/", 1)

# scale pixels in the range [-1, 1]
trainA = 2.0*(trainA/255.0)-1
trainB = 2.0*(trainB/255.0)-1

dataset = [trainA, trainB]
train(discriminator_A, discriminator_B, generator_A2B, generator_B2A, cycle_A2B, cycle_B2A, dataset)