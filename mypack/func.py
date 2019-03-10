import tensorflow as tf
import numpy as np

def one_hot_encoder(vec,vals):
	n  = len(vec)
	one_hot = np.zeros((n,vals))
	one_hot[range(n),vec] = 1

	return one_hot


def init_wts(shape):
	w = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(w)


def init_bias(shape):
    b = tf.constant(0.1,shape=shape)
    
    return tf.Variable(b)


def conv2d(x,W,strd,pad):
	return tf.nn.conv2d(x,W,strides = [1,strd[0],strd[1],1],padding=pad)


def max_pool(input_x,fil,strd,pad):
	return tf.nn.max_pool(input_x,ksize=[1,fil[0],fil[1],1],strides = [1,strd[0],strd[1],1],padding=pad)

def convolutional_layer(input_image,size_curr,strd,pad):

	w = init_wts(size_curr)
	b = init_bias([size_curr[3]])

	return tf.nn.relu(conv2d(input_image,w,strd,pad)+b)

def normal_full_layer(input_layer,size_next):
	input_size = int(input_layer.get_shape()[1])
	w = init_wts([input_size,size_next])
	b = init_wts([size_next])

	return tf.nn.relu(tf.matmul(input_layer,w)+b)



def create_placeholder(shape,n_cls):
	x = tf.placeholder(tf.float32,shape = [None,shape[0],shape[1],shape[2]])
	y_true = tf.placeholder(tf.float32,shape = [None,n_cls])
	return x,y_true


def incep_layer(x_image,I_LAYER_TYPE,I_LAYER_FIELD,NAME):
	mod_lis = []
	with tf.variable_scope(NAME) as scope:
		print('---------------------------------------------------------------------------------------')
		for i,k in enumerate(I_LAYER_TYPE):
			if k == 'CNN':
				nc = x_image.get_shape()[3]
				lst  = [I_LAYER_FIELD[i][0][0],I_LAYER_FIELD[i][0][1],int(nc),I_LAYER_FIELD[i][0][2]]
				mod_lis.append(convolutional_layer(x_image,lst,I_LAYER_FIELD[i][1],I_LAYER_FIELD[i][2]))
				print(mod_lis[-1])
			elif k == 'POOL':
				mod_lis.append(max_pool(x_image,I_LAYER_FIELD[i][0],I_LAYER_FIELD[i][1],I_LAYER_FIELD[i][2]))
				print(mod_lis[-1])
			elif k == 'CNN-CNN':
				nc = x_image.get_shape()[3]
				lst  = [I_LAYER_FIELD[i][0][0],I_LAYER_FIELD[i][0][1],int(nc),I_LAYER_FIELD[i][0][2]]
				temp = convolutional_layer(x_image,lst,I_LAYER_FIELD[i][1],I_LAYER_FIELD[i][2])
				nc = temp.get_shape()[3]
				lst  = [I_LAYER_FIELD[i][3][0],I_LAYER_FIELD[i][3][1],int(nc),I_LAYER_FIELD[i][3][2]]
				mod_lis.append(convolutional_layer(temp,lst,I_LAYER_FIELD[i][4],I_LAYER_FIELD[i][5]))
				print(mod_lis[-1])
			elif k == 'POOL-CNN':
				temp = max_pool(x_image,I_LAYER_FIELD[i][0],I_LAYER_FIELD[i][1],I_LAYER_FIELD[i][2])
				nc = temp.get_shape()[3]
				lst  = [I_LAYER_FIELD[i][3][0],I_LAYER_FIELD[i][3][1],int(nc),I_LAYER_FIELD[i][3][2]]
				mod_lis.append(convolutional_layer(temp,lst,I_LAYER_FIELD[i][4],I_LAYER_FIELD[i][5]))
				print(mod_lis[-1])
			print('---------------------------------------------------------')
	return tf.concat(mod_lis,axis=3,name=NAME)
