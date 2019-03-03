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

