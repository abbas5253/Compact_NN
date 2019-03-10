from impl_of_func import *
import tensorflow as tf
i1 = ['CNN','CNN-CNN','CNN-CNN','POOL-CNN']
i2 = [[[1,1,64],[1,1],'SAME'],[[1,1,96],[1,1],'SAME',[3,3,128],[1,1],'SAME'],[[1,1,16],[1,1],'SAME',[5,5,32],[1,1],'SAME'],[[3,3],[1,1],'SAME',[1,1,32],[1,1],'SAME']]
i3 = 'A1'


"""
CNN-CNN DONE
CNN-POOL undone
pool-cnn undone
cnn-pool undone
"""


NN = NeuralNet([None,748],[None,10])
train,y_preds,cost = NN.CompactNeuralNetwork(NN.x,NN.y_true,
                                          hidden_layers=4,
                                          layer_type=['CNN','POOL','IN','FC'],
                                          layer_fields=[[[5,5,1,64],[1,1],'SAME'],[[3,3],[2,2],'VALID'],[i1,i2,i3],[10]],
                                          Optimization='Adam',
                                          learning_rate = 0.01)