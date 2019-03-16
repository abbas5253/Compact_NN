from impl_of_func import *
import tensorflow as tf
i1 = ['CNN','CNN-CNN','CNN-CNN','POOL-CNN']
i2 = [[[1,1,64],[1,1],'SAME'],[[1,1,96],[1,1],'SAME',[3,3,128],[1,1],'SAME'],[[1,1,16],[1,1],'SAME',[5,5,32],[1,1],'SAME'],[[3,3],[1,1],'SAME',[1,1,32],[1,1],'SAME']]
i3 = 'A1'

i2_2 = [[[1,1,128],[1,1],'SAME'],[[1,1,128],[1,1],'SAME',[3,3,192],[1,1],'SAME'],[[1,1,32],[1,1],'SAME',[5,5,96],[1,1],'SAME'],[[3,3],[1,1],'SAME',[1,1,64],[1,1],'SAME']]

i2_3 = [[[1,1,192],[1,1],'SAME'],[[1,1,96],[1,1],'SAME',[3,3,208],[1,1],'SAME'],[[1,1,16],[1,1],'SAME',[5,5,48],[1,1],'SAME'],[[3,3],[1,1],'SAME',[1,1,64],[1,1],'SAME']]


i2_4 = [[[1,1,160],[1,1],'SAME'],[[1,1,112],[1,1],'SAME',[3,3,224],[1,1],'SAME'],[[1,1,24],[1,1],'SAME',[5,5,64],[1,1],'SAME'],[[3,3],[1,1],'SAME',[1,1,64],[1,1],'SAME']]


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



NN = NeuralNet([None,784],[None,10])
train,y_preds,cost = NN.CompactNeuralNetwork(NN.x,NN.y_true,
                                          hidden_layers=9,
                                          layer_type=['CNN','POOL','IN','IN','POOL','IN','IN','APOOL','FC-False'],
                                          layer_fields=[[[5,5,1,64],[1,1],'SAME'],[[3,3],[2,2],'SAME'],[i1,i2,i3],[i1,i2_2,'A2'],[[3,3],[2,2],'VALID'],[i1,i2_3,'A3'],[i1,i2_4,'A4'],[[6,6],[1,1],'VALID'],[10]],
                                          Optimization='Adam',
                                          learning_rate = 0.01)


init = tf.global_variables_initializer()
cor = tf.equal(tf.argmax(y_preds,1),tf.argmax(NN.y_true,1))

acc = tf.reduce_mean(tf.cast(cor,tf.float32))

steps = 500
with tf.Session() as sess:
  sess.run(init)
    
  for i in range(steps):
      
        
    batch_x, batch_y = mnist.train.next_batch(128)
    sess.run(train,feed_dict={NN.x:batch_x,NN.y_true:batch_y})
        
    if i%10 == 0:
    
      train_acc = sess.run(acc,feed_dict={NN.x:batch_x,NN.y_true:batch_y})
      test_acc = sess.run(acc,feed_dict={NN.x:mnist.test.images[:1000],NN.y_true:mnist.test.labels[:1000]})
    
      print(i,train_acc,test_acc)

      