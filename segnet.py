from keras.models import *
from keras.layers import *
from .basic_encoder import basic_encoder
from .vgg16 import get_vgg_encoder
from .resnet50 import get_resnet50_encoder

def segnet_decoder(x, n_classes, n_up=3):

	assert n_up >= 2

	x = ZeroPadding2D((1,1))(x)
	x = Conv2D(512, (3,3), padding='valid')(x)
	x = BatchNormalization()(x)
	x = UpSampling2D((2,2))(x)
	x = ZeroPadding2D((1,1))(x)
	x = Conv2D(256, (3, 3), padding='valid')(x)
	x = BatchNormalization()(x)

	for _ in range(n_up-2):
		x = UpSampling2D((2,2))(x)
		x = ZeroPadding2D((1,1))(x)
		x = Conv2D(128, (3,3), padding='valid')(x)
		x = BatchNormalization()(x)

	x = UpSampling2D((2,2))(x)
	x = ZeroPadding2D((1,1))(x)
	x = Conv2D(64, (3,3), padding='valid')(x)
	x = BatchNormalization()(x)
	x = Conv2D(n_classes, (3,3), padding='same')(x)
	return x

def segnet(n_classes, backbone='basic_encoder', input_height=416, input_width=608):

	encoder = basic_encoder
	if backbone == 'basic_encoder':
		encoder = basic_encoder
	elif backbone == 'vgg16':
		encoder = get_vgg_encoder
	elif backbone == 'resnet50':
		encoder = get_resnet50_encoder
	
	img_input, levels = encoder(input_height=input_height, input_width=input_width)
	feat = levels[3]
	output = segnet_decoder(feat, n_classes, n_up=4)
	output = Activation('sigmoid')(output)
	model = Model(img_input, output)

	return model