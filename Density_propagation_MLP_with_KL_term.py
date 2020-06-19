import tensorflow as tf
from tensorflow import keras
import os
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
#from Adding_noise import random_noise

plt.ioff()

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def x_Sigma_w_x_T(x, W_Sigma):
  batch_sz = x.shape[0]
  xx_t = tf.reduce_sum(tf.multiply(x, x),axis=1, keepdims=True)               
  xx_t_e = tf.expand_dims(xx_t,axis=2)                                     
  return tf.multiply(xx_t_e, W_Sigma)

def w_t_Sigma_i_w (w_mu, in_Sigma):
  Sigma_1_1 = tf.matmul(tf.transpose(w_mu), in_Sigma)
  Sigma_1_2 = tf.matmul(Sigma_1_1, w_mu)
  return Sigma_1_2

def tr_Sigma_w_Sigma_in (in_Sigma, W_Sigma):
  Sigma_3_1 = tf.linalg.trace(in_Sigma)
  Sigma_3_2 = tf.expand_dims(Sigma_3_1, axis=1)
  Sigma_3_3 = tf.expand_dims(Sigma_3_2, axis=1)
  return tf.multiply(Sigma_3_3, W_Sigma) 

def activation_Sigma (gradi, Sigma_in):
  grad1 = tf.expand_dims(gradi,axis=2)
  grad2 = tf.expand_dims(gradi,axis=1)
  return tf.multiply(Sigma_in, tf.matmul(grad1, grad2))
 



# Linear Class - First Layer (Constant * RV)
class LinearFirst(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units):
        super(LinearFirst, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w_mu = self.add_weight(name='w_mu',
            shape=(input_shape[-1], self.units),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
            trainable=True
        )
        self.w_sigma = self.add_weight(name='w_sigma',
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(minval= -12., maxval=-2.2, seed=None),
            trainable=True
        )
        self.b_mu = self.add_weight(name='b_mu',
            shape=(self.units,), initializer=tf.random_normal_initializer( mean=0.0, stddev=0.00005, seed=None),
            trainable=True
        )
        self.b_sigma = self.add_weight(name='b_sigma',
            shape=(self.units,), initializer=tf.random_uniform_initializer(minval= -12., maxval=-10., seed=None),
            trainable=True
        )
    def call(self, inputs):
        # Mean
        #print(self.w_mu.shape)
        mu_out = tf.matmul(inputs, self.w_mu) + self.b_mu                         # Mean of the output
        # Varinace
        W_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.w_sigma)))                                        
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma) + tf.math.log(1. + tf.math.exp(self.b_sigma)) 

        Term1 = self.w_mu.shape[0]*tf.math.log(tf.math.log(1. + tf.math.exp(self.w_sigma)))
        Term2 = tf.reduce_sum(tf.reduce_sum(tf.abs(self.w_mu)))
        Term3 = self.w_mu.shape[0]*tf.math.log(1. + tf.math.exp(self.w_sigma))      
      
        kl_loss = -0.5 * tf.reduce_mean(Term1 - Term2 - Term3)
        self.add_loss(kl_loss)

        return mu_out, Sigma_out


# Linear Class - Second Layer (RV * RV)
class LinearNotFirst(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units):
        super(LinearNotFirst, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w_mu = self.add_weight(name='w_mu',
            shape=(input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None),
            trainable=True,
        )
        self.w_sigma = self.add_weight(name='w_sigma',
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(minval= -12., maxval=-2.2, seed=None),
            trainable=True,
        )
        self.b_mu = self.add_weight(name='b_mu',
            shape=(self.units,), initializer=tf.random_normal_initializer( mean=0.0, stddev=0.00005, seed=None),
            trainable=True,
        )
        self.b_sigma = self.add_weight(name='b_sigma',
            shape=(self.units,), initializer=tf.random_uniform_initializer(minval= -12., maxval=-10., seed=None),
            trainable=True,
        )

    def call(self, mu_in, Sigma_in):
        
        mu_out = tf.matmul(mu_in, self.w_mu) + self.b_mu    
        W_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.w_sigma)))                                   
        # Simga_out has three terms
        Sigma_1 = w_t_Sigma_i_w (self.w_mu, Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)                                   
        Sigma_3 = tr_Sigma_w_Sigma_in (Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3 + tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.b_sigma))) 
        
        Term1 = self.w_mu.shape[0]*tf.math.log(tf.math.log(1. + tf.math.exp(self.w_sigma)))
        Term2 = tf.math.reduce_sum(tf.reduce_sum(tf.abs(self.w_mu)))
        Term3 = self.w_mu.shape[0]*tf.math.log(1. + tf.math.exp(self.w_sigma))    
        kl_loss = -0.5 * tf.reduce_mean(Term1 - Term2 - Term3)
        self.add_loss(kl_loss)
        return mu_out, Sigma_out

class myReLU(keras.layers.Layer):
    """ReLU"""

    def __init__(self):
        super(myReLU, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        with tf.GradientTape() as g:
          g.watch(mu_in)
          out = tf.nn.relu(mu_in)
        gradi = g.gradient(out, mu_in) 

        Sigma_out = activation_Sigma (gradi, Sigma_in)
        
        return mu_out, Sigma_out

class mysoftmax(keras.layers.Layer):
    """Mysoftmax"""

    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.softmax(mu_in)
        pp1 = tf.expand_dims(mu_out, axis=2)
        pp2 = tf.expand_dims(mu_out, axis=1)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        Sigma_out = tf.matmul(grad, tf.matmul(Sigma_in, tf.transpose(grad, perm=[0, 2, 1])))
        return mu_out, Sigma_out


def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels=2, batch_size=200):
    NS = tf.linalg.diag(tf.constant(1e-3, shape=[batch_size, num_labels]))
    I = tf.eye(num_labels, batch_shape=[batch_size])
    y_pred_sd_ns = y_pred_sd + NS
    y_pred_sd_inv = tf.linalg.solve(y_pred_sd_ns, I)
    mu_ = y_pred_mean - y_test
    mu_sigma = tf.matmul(mu_ ,  y_pred_sd_inv) 
    ms = 0.5*tf.matmul(mu_sigma , mu_, transpose_b=True) + 0.5*tf.linalg.slogdet(y_pred_sd_ns)[1]
    ms = tf.reduce_mean(ms)
    return(ms)

class exVDPMLP(tf.keras.Model):
    """Stack of Linear layers with a KL regularization loss."""

    def __init__(self, name=None):
        super(exVDPMLP, self).__init__()
        self.linear_1 = LinearFirst(256)
        self.myrelu_1 = myReLU()
        self.linear_2 = LinearNotFirst(2)
        self.mysoftma = mysoftmax()

    def call(self, inputs):
        m, s = self.linear_1 (inputs)
        m, s = self.myrelu_1 (m, s)
        m, s = self.linear_2 (m, s)
        outputs, Sigma = self.mysoftma(m, s)        
        return outputs, Sigma

def main_function( input_dim = 10, units = 256, output_size = 2 , batch_size = 200, epochs = 20, lr = 0.001, 
         Random_noise=True, gaussain_noise_std=1, Training = False):
    

    PATH = './saved_models/DP_MLP_epoch_{}/'.format(epochs)
    X = np.load('./x_train.npy')
    y = np.load('./y_train.npy')
    x_train = X[0:120000,:]
    y_train = y[0:120000]
    x_test = X[120000:133600,:]
    y_test = y[120000:133600]   

    one_hot_y_train = tf.one_hot(y_train.astype(np.float32), depth=2)
    one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=2) 

    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)

        
    # Cutom Trianing Loop with Graph
    mlp_model = exVDPMLP(name='mlp')    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits, sigma = mlp_model(x)      
            loss_final = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+10),
                                       clip_value_max=tf.constant(1e+10)), output_size , batch_size)
            loss_layers = sum(mlp_model.losses)

            loss = loss_final + 0.001*loss_layers
            gradients = tape.gradient(loss, mlp_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mlp_model.trainable_weights))        
        return loss, logits
    if Training:
        train_acc = np.zeros(epochs) 
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        
        for epoch in range(epochs):
          print('Epoch: ', epoch+1, '/' , epochs)
    
          acc1 = 0 
          acc_valid1 = 0 
          err1 = 0
          err_valid1 = 0
          tr_no_steps = 0          
          #Training
          for step, (x, y) in enumerate(tr_dataset):
              update_progress(step / int(x_train.shape[0] / (batch_size)) )  
              loss, logits = train_on_batch(x, y)
              err1+= loss
    
              corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
              accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
              acc1+=accuracy                 
              tr_no_steps+=1
          train_acc[epoch] = acc1/tr_no_steps
          train_err[epoch] = err1/tr_no_steps
          
          print('Training Acc  ', train_acc[epoch])
          print('Training error  ', train_err[epoch])          
          
        stop = timeit.default_timer()
        print('Total Training Time: ', stop - start)
        print('Training Acc  ', np.mean(train_acc))          
        print('Training error  ', np.mean(train_err))       
    
        
        mlp_model.save_weights(PATH + 'DP_MLP_model')
        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')            
            plt.ylim(0, 1.1)
            plt.title("Density Propagation MLP on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'DP_MLP_on_MNIST_Data_acc.png')
            plt.close(fig)
    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')            
            plt.title("Density Propagation MLP on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'DP_MLP_on_MNIST_Data_error.png')
            plt.close(fig)
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')         
        pickle.dump([train_acc, train_err], f)                                                   
        f.close()         
             
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr))            
        textfile.write("\n---------------------------------")
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))                  
                textfile.write("\n Averaged Training  error : "+ str( train_err))               
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc)))               
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err)))               
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
        
    else:
        test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        mlp_model.load_weights(PATH + 'DP_MLP_model')
        test_no_steps = 0
        err_test = 0
        acc_test = 0
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size,  input_dim])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size])
        logits_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, output_size, output_size])
        for step, (x, y) in enumerate(val_dataset):
          update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
          true_x[test_no_steps, :, :] = x
          true_y[test_no_steps, :, :] = y
          if Random_noise:
              noise = tf.random.normal(shape = [batch_size, input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
              x = x +  noise 

          loss_layers = sum(mlp_model.losses)       
          logits, sigma = mlp_model(x)  
          logits_[test_no_steps,:,:] =logits
          sigma_[test_no_steps, :, :, :]= sigma
          tloss = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+10), clip_value_max=tf.constant(1e+10)), output_size , batch_size)+ 0.001*loss_layers
          err_test+= tloss
          
          corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
          accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
          acc_test+=accuracy

          if step % 500 == 0:
              print("Step:", step, "Loss:", float(tloss))
              print("Total running accuracy so far: %.3f" % accuracy)              
           
          test_no_steps+=1
       
        test_acc = acc_test/test_no_steps      
        test_error = err_test/test_no_steps
        print('Test accuracy : ', test_acc.numpy())
        print('Test error : ', test_error.numpy())
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_, sigma_, true_x, true_y, test_acc.numpy(), test_error.numpy()  ], pf)                                                   
        pf.close()
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr))          
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc.numpy()))
        textfile.write("\n Averaged Test error : "+ str(test_error.numpy() ))            
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))              
        textfile.write("\n---------------------------------")    
        textfile.close()
            
if __name__ == '__main__':
    main_function()    
