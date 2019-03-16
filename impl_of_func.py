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
				incep_layers = input("Inception_layer {}".format(i+1))
				
				i_l = wg.Text(description='Inx=ception_layer{}'.format(i+1),value = incep_layers)
				incep_layers = list(incep_layers.split('-'))
				self.feilds[i].append(i_l)
				for layer in incep_layers:
					if layer == 'CNN':
						filters = wg.Text(description='filter of {} {}'.format(i+1,layer))
						strides = wg.Text(description= 'Strd of {} {}'.format(i+1,layer))
						pad = wg.Text(description= 'Padding{} {}'.format(i+1,layer))
						self.feilds[i].append(filters)
						self.feilds[i].append(strides)
						self.feilds[i].append(pad)
					elif layer == 'POOL':
						filters = wg.Text(description='filter of {} {}'.format(i+1,layer))
						strides = wg.Text(description= 'Strd of {} {}'.format(i+1,layer))
						pad = wg.Text(description= 'Padding{} {}'.format(i+1,layer))
						self.feilds[i].append(filters)
						self.feilds[i].append(strides)
						self.feilds[i].append(pad)
					elif layer == 'CNN>CNN':
						filters1 = wg.Text(description='1filter of {} {}'.format(i+1,layer))
						strides1 = wg.Text(description= '1Strd of {} {}'.format(i+1,layer))
						pad1 = wg.Text(description= '1Padding{} {}'.format(i+1,layer))
						self.feilds[i].append(filters1)
						self.feilds[i].append(strides1)
						self.feilds[i].append(pad1)
						filters2 = wg.Text(description='2filter of {} {}'.format(i+1,layer))
						strides2 = wg.Text(description= '2Strd of {} {}'.format(i+1,layer))
						pad2 = wg.Text(description= '2Padding{} {}'.format(i+1,layer))
						self.feilds[i].append(filters2)
						self.feilds[i].append(strides2)
						self.feilds[i].append(pad2)
					elif layer == 'POOL>CNN':
						filters1 = wg.Text(description='1filter of {} {}'.format(i+1,layer))
						strides1 = wg.Text(description= '1Strd of {} {}'.format(i+1,layer))
						pad1 = wg.Text(description= '1Padding{} {}'.format(i+1,layer))
						self.feilds[i].append(filters1)
						self.feilds[i].append(strides1)
						self.feilds[i].append(pad1)
						filters2 = wg.Text(description='2filter of {} {}'.format(i+1,layer))
						strides2 = wg.Text(description= '2Strd of {} {}'.format(i+1,layer))
						pad2 = wg.Text(description= '2Padding{} {}'.format(i+1,layer))
						self.feilds[i].append(filters2)
						self.feilds[i].append(strides2)
						self.feilds[i].append(pad2)


			else:
				Size = wg.Text(description= 'Size{}'.format(i+1))
				self.feilds[i].append(Size)


            
		for i in range(len(self.feilds)):
			for j in range(len(self.feilds[i])):
				display(self.feilds[i][j])
            
	def set_input_layer_feilds(self):
		self.layer_feilds = [[self.feilds[i][j].value for j in range(len(self.feilds[i]))] for i in range(len(self.feilds))]
		for i in range(len(self.layer_type)):
			if self.layer_type[i] == 'CNN':
				self.layer_feilds[i] = [[int(k) for k in self.layer_feilds[i][0].split(',')]]+[[int(k) for k in self.layer_feilds[i][1].split(',')]]+[self.layer_feilds[i][2]]
			elif self.layer_type[i] == 'POOL':
				self.layer_feilds[i] = [[int(k) for k in self.layer_feilds[i][0].split(',')]]+[[int(k) for k in self.layer_feilds[i][1].split(',')]]+[self.layer_feilds[i][2]]
			



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

			elif layer_type[layer] == 'FC-False':

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




