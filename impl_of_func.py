from mypack import func
import numpy as np
import tensorflow as tf
from functools import reduce

import ipywidgets as wg
from IPython.display import display



class NeuralNet:


	def __init__(self,shape_of_input,shape_of_output,no_hidden_layers):
		self.x = tf.placeholder(tf.float32,shape=shape_of_input)
		self.y_true = tf.placeholder(tf.float32,shape= shape_of_output)
		self.no_hidden_layers = no_hidden_layers
		self.hold_prob = tf.placeholder(tf.float32)
		self.k = None
		self.layer_type=  None
		self.feilds = None
		self.layer_feilds = [[] for _ in range(no_hidden_layers)]

	def input_layer_type(self):
		self.k= wg.Text(description='Layer Type')
		display(self.k)

	def set_input_layer(self):
		l_t = (self.k).value.split('-')
		self.layer_type = l_t

	def input_layer_feilds(self):
		layer_type = self.layer_type
		self.feilds= [[] for _ in range(len(layer_type))]
		for i in range(len(layer_type)):
			if layer_type[i] == 'CNN':
				filters = wg.Text(description='filter{}'.format(i+1))
				strides = wg.Text(description= 'Strd{}'.format(i+1))
				pad = wg.Text(description= 'Padding{}'.format(i+1))
				self.feilds[i].append(filters)
				self.feilds[i].append(strides)
				self.feilds[i].append(pad)
			elif layer_type[i] == 'POOL':
				filters = wg.Text(description='filter{}'.format(i+1))
				strides = wg.Text(description= 'Strd{}'.format(i+1))
				pad = wg.Text(description= 'Padding{}'.format(i+1))
				self.feilds[i].append(filters)
				self.feilds[i].append(strides)
				self.feilds[i].append(pad)
			elif layer_type[i] == 'IN':
				incep_layer = wg.Text(description='incep_layer_type for incep{}'.format(i+1))
				self.feilds[i].append(incep_layer)
			else:
				Size = wg.Text(description= 'Size{}'.format(i+1))
				self.feilds[i].append(Size)


            
		for i in range(len(self.feilds)):
			for j in range(len(self.feilds[i])):
				display(self.feilds[i][j])
            
	def set_input_layer_feilds(self):
		self.layer_feilds = [[self.feilds[i][j].value  for j in range(len(self.feilds[i]))] for i in range(len(self.feilds))]
		for i in range(len(self.layer_type)):
			if self.layer_type[i] == 'CNN':
				self.layer_feilds[i] = [[int(k) for k in self.layer_feilds[i][0].split(',')]]+[[int(k) for k in self.layer_feilds[i][1].split(',')]]+[self.layer_feilds[i][2]]
			elif self.layer_type[i] == 'POOL':
				self.layer_feilds[i] = [[int(k) for k in self.layer_feilds[i][0].split(',')]]+[[int(k) for k in self.layer_feilds[i][1].split(',')]]+[self.layer_feilds[i][2]]
			elif self.layer_type[i] == 'IN':
				self.layer_feilds[i] = [self.layer_feilds[i][0].split(',')]
				tem = []
				incep_layer = self.layer_feilds[i][0]
				for layer in incep_layer:
					if layer  == 'CNN':
						f_ = input('filter for layer inceptions {} '.format(layer)).split(',')
						f_ = [int(k) for k in f_]
						s_ = input('Strides for layer inceptions {}'.format(layer)).split(',')
						s_ = [int(k) for k in s_]
						p_ = input('Padding for layer inceptions {}'.format(layer))
						tem.append([f_,s_,p_])
					elif layer == 'CNN-CNN':
						f_1 = input('filter for layer inceptions{} '.format(layer)).split(',')
						f_1 = [int(k) for k in f_1]
						s_1= input('Strides for layer inceptions{} '.format(layer)).split(',')
						s_1 = [int(k) for k in s_1]
						p_1 = input('Padding for layer inceptions{} '.format(layer))
						f_2 = input('filter for layer inceptions{} '.format(layer)).split(',')
						f_2 = [int(k) for k in f_2]
						s_2 = input('Strides for layer inceptions{} '.format(layer)).split(',')
						s_2 = [int(k) for k in s_2]
						p_2 = input('Padding for layer inceptions{} '.format(layer))
						tem.append([f_1,s_1,p_1,f_2,s_2,p_2])
					elif layer == 'POOL-CNN':
						f_11 = input('filter for layer inceptions{} '.format(layer)).split(',')
						f_11 = [int(k) for k in f_11]
						s_11= input('Strides for layer inceptions{} '.format(layer)).split(',')
						s_11 = [int(k) for k in s_11]
						p_11 = input('Padding for layer inceptions{} '.format(layer))
						f_22 = input('filter for layer inceptions{} '.format(layer)).split(',')
						f_22 = [int(k) for k in f_22]
						s_22 = input('Strides for layer inceptions{} '.format(layer)).split(',')
						s_22 = [int(k) for k in s_22]
						p_22 = input('Padding for layer inceptions{} '.format(layer))
						tem.append([f_11,s_11,p_11,f_22,s_22,p_22])
					self.layer_feilds[i].append(tem)
				self.layer_feilds[i].append('a'+str(i))


			else:
				self.layer_feilds[i] = [int(self.layer_feilds[i][0])]


	def CompactNeuralNetwork(self,x_in,y_true,hidden_layers=6,layer_type=['CNN','POOL','CNN','POOL','FC','FC'],layer_fields=[[[5,5,1,32],[1,1],'SAME'],[[2,2],[2,2],'SAME'],[[5,5,32,64],[1,1],'SAME'],[[2,2],[2,2],'SAME'],[512],[10]],hold_prob=0.4,Optimization='Adam',learning_rate = 0.01):

		print('##########################################################################################')

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
				print(X_cur)

			elif layer_type[layer] == 'APOOL':

				k_size = layer_fields[layer][0]
				strides = layer_fields[layer][1]
				pad = layer_fields[layer][2]

				X_cur  = func.avg_pool(X_cur,list(k_size),list(strides),pad)
				print(X_cur)



			elif layer_type[layer] == 'FC':

				s = [int(f) for f in X_cur.get_shape()[1:]]
			
				dim = reduce(lambda x, y: x*y,s)

				X_cur = tf.reshape(X_cur,[-1,dim])

				sizecur = layer_fields[layer][0]

				X_cur  = func.normal_full_layer(X_cur,int(sizecur))
				print(X_cur)

			elif layer_type[layer] == 'FC_False':

				s = [int(f) for f in X_cur.get_shape()[1:]]
			
				dim = reduce(lambda x, y: x*y,s)

				X_cur = tf.reshape(X_cur,[-1,dim])

				sizecur = layer_fields[layer][0]

				X_cur  = func.normal_full_layer(X_cur,int(sizecur),relu=False)
				print(X_cur)

			elif layer_type[layer] == 'FC_Dropout':
				s = [int(f) for f in X_cur.get_shape()[1:]]
			
				dim = reduce(lambda x, y: x*y,s)

				X_cur = tf.reshape(X_cur,[-1,dim])

				sizecur = layer_fields[layer][0]

				X_cur  = func.normal_full_layer(X_cur,int(sizecur))

				X_cur = tf.nn.dropout(X_cur,keep_prob=hold_prob)
				print(X_cur)

			elif layer_type[layer] == 'IN':
				
				X_cur= func.incep_layer(X_cur,layer_fields[layer][0],layer_fields[layer][1],layer_fields[layer][2])
				print(X_cur)

		print('###################################################################################################################3')    
           
		y_preds = X_cur

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_preds,labels= y_true))

		if Optimization == 'Adam':

			optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

		elif Optimization == 'GradinentDescent':

			optimizer = tf.train.GradinentDescentOptimizer(learning_rate= learning_rate)


		train = optimizer.minimize(cross_entropy)



		return train,y_preds,cross_entropy




