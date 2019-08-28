from keras.layers import *

def basic_encoder(input_height=224, input_width=224  ):

	kernel, filter_size, pad, pool_size = 3, 64, 1, 2

	img_input = Input(shape=(input_height,input_width , 3 ))
	x = img_input
	levels = []

	x = ZeroPadding2D((pad,pad))(x)
	x = Conv2D(filter_size, (kernel, kernel), padding='valid')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')( x )
	x = MaxPooling2D((pool_size, pool_size))(x)
	levels.append(x)

	x = ZeroPadding2D((pad,pad))(x)
	x = Conv2D(128, (kernel, kernel) ,padding='valid')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((pool_size, pool_size))(x)
	levels.append(x)

	for _ in range(3):
		x = ZeroPadding2D((pad,pad))(x)
		x = Conv2D(256, (kernel, kernel) , padding='valid')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D((pool_size, pool_size))(x)
		levels.append(x)

	return img_input , levels




