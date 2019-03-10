from mypack import func
import numpy as np
import tensorflow as tf
from functools import reduce


class NeuralNet:


	def __init__(self,shape_of_input,shape_of_output):
		self.x = tf.placeholder(tf.float32,shape=shape_of_input)
		self.y_true = tf.placeholder(tf.float32,shape= shape_of_output)

	def CompactNeuralNetwork(self,x_in,y_true,hidden_layers=6,layer_type=['CNN','POOL','CNN','POOL','FC','FC'],layer_fields=[[[5,5,1,32],[1,1],'SAME'],[[2,2],[2,2],'SAME'],[[5,5,32,64],[1,1],'SAME'],[[2,2],[2,2],'SAME'],[512],[10]],Optimization='Adam',learning_rate = 0.01):
	

		layer_type = np.array(layer_type).reshape(hidden_layers,1)
		layer_fields = np.array(layer_fields).reshape(hidden_layers,)	
		X_cur =  tf.reshape(x_in,[-1,28,28,1])

		for layer in range(hidden_layers):

			if layer_type[layer] == 'CNN':

				filters = layer_fields[layer][0]
				strd = layer_fields[layer][1]
				pad = layer_fields[layer][2]

				X_cur = func.convolutional_layer(X_cur,list(filters),strd= list(strd),pad= pad)


			elif layer_type[layer] == 'POOL':

				k_size = layer_fields[layer][0]
				strides = layer_fields[layer][1]
				pad = layer_fields[layer][2]

				X_cur  = func.max_pool(X_cur,list(k_size),list(strides),pad)


			elif layer_type[layer] == 'FC':

				s = [int(f) for f in X_cur.get_shape()[1:]]
			
				dim = reduce(lambda x, y: x*y,s)

				X_cur = tf.reshape(X_cur,[-1,dim])

				sizecur = layer_fields[layer][0]

				X_cur  = func.normal_full_layer(X_cur,int(sizecur))
			elif layer_type[layer] == 'IN':
				
				X_cur= func.incep_layer(X_cur,layer_fields[layer][0],layer_fields[layer][1],layer_fields[layer][2])

                
           
		y_preds = X_cur

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_preds,labels= y_true))

		if Optimization == 'Adam':

			optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

		elif Optimization == 'GradinentDescent':

			optimizer = tf.train.GradinentDescentOptimizer(learning_rate= learning_rate)


		train = optimizer.minimize(cross_entropy)



		return train,y_preds,cross_entropy




