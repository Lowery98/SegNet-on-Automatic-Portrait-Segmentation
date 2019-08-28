from keras.models import *
from keras.layers import *
from .basic_encoder import basic_encoder
from .vgg16 import get_vgg_encoder
from .resnet50 import get_resnet50_encoder

def crop(o1, o2, i):

	o_shape1 = Model(i, o1).output_shape
	output_height1 = o_shape1[1]
	output_width1 = o_shape1[2]

	o_shape2 = Model(i, o2).output_shape
	output_height2 = o_shape2[1]
	output_width2 = o_shape2[2]

	cx = abs( output_width1 - output_width2 )
	cy = abs( output_height2 - output_height1 )

	if output_width1 > output_width2:
		o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
	else:
		o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)
	
	if output_height1 > output_height2 :
		o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
	else:
		o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

	return o1 , o2 

def fcn(n_classes, backbone='basic_encoder', input_height=416, input_width=608):

	encoder = basic_encoder
	if backbone == 'basic_encoder':
		encoder = basic_encoder
	elif backbone == 'vgg16':
		encoder = get_vgg_encoder
	elif backbone == 'resnet50':
		encoder = get_resnet50_encoder

	img_input , levels = encoder(input_height=input_height, input_width=input_width)
	[_, _, f3, f4, f5] = levels

	o = f5
	o = Conv2D(512, (7, 7), activation='relu', padding='same')(o)
	o = Dropout(0.5)(o)
	o = Conv2D(512, (1, 1), activation='relu', padding='same')(o)
	o = Dropout(0.5)(o)

	o = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(o)
	o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(o)

	o2 = f4
	o2 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(o2)
	
	o, o2 = crop(o, o2, img_input)
	
	o = Add()([o, o2])

	o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(o)
	o2 = f3 
	o2 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(o2)
	o2, o = crop(o2, o, img_input)
	o = Add()([o2, o])

	o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False, padding='same')(o)
	o = Activation('sigmoid')(o)
	
	model = Model(img_input , o )
	return model