import tensorflow as tf
import numpy as np

import sys, os

tf.compat.v1.disable_eager_execution()

from inspect import isgenerator

from regularizations import Regularizer
from loss_functions import LossFunction
from initializations import WeightInitializer, GaussianSpectralRadius


def relu(x):
    return tf.nn.relu(x)


def gelu(x):
    return tf.nn.gelu(x)


def gelu_0_2(x):
    return 0.2 + tf.nn.gelu(x)




class RNN(object):
    
    
    def __init__(self, params):
        
        self.params = params
        
        # RNN name
        try:
            self.name = params['name']
        except KeyError:
            print('Missing identifier name for RNN!')
            raise
            
        # required inputs
        try:
            self.N_in = params['N_in']
        except KeyError:
            print("Unspecified N_in")
            raise
        try:
            self.N_rec = params['N_rec']
        except KeyError:
            print("Unspecified N_rec")
            raise
        try:
            self.N_out = params['N_out']
        except KeyError:
            print("Unspecified N_out")
            raise
        try:
            self.N_steps = params['N_steps']
        except KeyError:
            print("Unspecified N_steps")
            raise
        try:
            self.dt = params['dt']
        except KeyError:
            print("Unspecified dt")
            raise    
            
        self.N_rec_multiply = params.get('N_rec_multiply', 1)
            
        # default tau is 100
        self.tau = params.get('tau', 100)
        
        # for euler integration
        self.alpha = (1.0*self.dt) / self.tau

        # default N_batch is 64
        self.N_batch = params.get('N_batch', 64)
            
        # default rec_noise is 0
        self.rec_noise = params.get('rec_noise', 0.)
        
        # set opto effect type
        self.opto_affect = params.get('opto_affect', None)
     
        mapping = {'relu': relu,
                   'gelu': gelu,
                   'gelu_0_2': gelu_0_2}    
        # default nonlinearity is relu
        self.transfer_function = mapping[params.get('transfer_function', 'relu')]
        
        # path for loading weights
        self.load_weights_path = params.get('load_weights_path', None)
        
        # weight initializations / specifications
        if self.load_weights_path is not None:
            self.initializer = WeightInitializer(load_weights_path = self.load_weights_path)
        else:
            self.initializer = params.get('initializer', GaussianSpectralRadius(**params))
            
        self.dale_ratio = self.initializer.get_dale_ratio()
        
        # mark trainable parameters:
        self.init_state_train = params.get('init_state_train', True)
        self.W_in_train = params.get('W_in_train', True)
        self.W_rec_train = params.get('W_rec_train', True)
        self.W_out_train = params.get('W_out_train', True)
        self.b_rec_train = params.get('b_rec_train', False)
        self.b_out_train = params.get('b_out_train', False)
        
        # tensorflow placeholders for input, output and mask 
        self.x = tf.compat.v1.placeholder("float", [None, self.N_steps, self.N_in])
        self.y = tf.compat.v1.placeholder("float", [None, self.N_steps, self.N_out])
        self.output_mask = tf.compat.v1.placeholder("float", [None, self.N_steps, self.N_out])
        
        
        # initialize tensorflow variables:
        with tf.compat.v1.variable_scope(self.name) as scope:
            
            # initial state of RNN units
            try:
                self.init_state = tf.compat.v1.get_variable('init_state', [1, self.N_rec],
                                                            initializer = self.initializer.get('init_state'),
                                                            trainable = self.init_state_train)
            except ValueError as error:
                raise UserWarning("tensorflow scope aka model name already exists")
                
            self.init_state = tf.tile(self.init_state, [self.N_batch, 1])
            
            # input weight matrix
            self.W_in = tf.compat.v1.get_variable('W_in', [self.N_rec, self.N_in],
                                                  initializer = self.initializer.get('W_in'),
                                                  trainable = self.W_in_train)
            
            # recurrent weight matrix
            self.W_rec = tf.compat.v1.get_variable('W_rec', [self.N_rec, self.N_rec],
                                                   initializer = self.initializer.get('W_rec'),
                                                   trainable = self.W_rec_train)
            
            # output weight matrix
            self.W_out = tf.compat.v1.get_variable('W_out', [self.N_out, self.N_rec],
                                                   initializer = self.initializer.get('W_out'),
                                                   trainable = self.W_out_train)
            
            # recurrent bias
            self.b_rec = tf.compat.v1.get_variable('b_rec', [self.N_rec],
                                                   initializer = self.initializer.get('b_rec'),
                                                   trainable = self.b_rec_train)
            
            # output bias
            self.b_out = tf.compat.v1.get_variable('b_out', [self.N_out],
                                                   initializer = self.initializer.get('b_out'),
                                                   trainable = self.b_out_train)
            
            
            
            ## NON TRAINABLE ONES:
            
            # dale's law for recurrent weights:
            self.Dale_rec = tf.compat.v1.get_variable('Dale_rec',
                                                      [self.N_rec, self.N_rec],
                                                      initializer = self.initializer.get('Dale_rec'),
                                                      trainable = False)
            
            # dale's law for output weights:
            self.Dale_out = tf.compat.v1.get_variable('Dale_out',
                                                      [self.N_rec, self.N_rec],
                                                      initializer = self.initializer.get('Dale_out'),
                                                      trainable = False)
            
            # connectivity matrices:
            self.input_connectivity = tf.compat.v1.get_variable('input_connectivity', 
                                                                [self.N_rec, self.N_in],
                                                                initializer = self.initializer.get('input_connectivity'),
                                                                trainable = False)
            
            self.rec_connectivity = tf.compat.v1.get_variable('rec_connectivity', 
                                                              [self.N_rec, self.N_rec],
                                                              initializer=self.initializer.get('rec_connectivity'),
                                                              trainable=False)
            
            self.output_connectivity = tf.compat.v1.get_variable('output_connectivity', 
                                                                 [self.N_out, self.N_rec],
                                                                initializer=self.initializer.get('output_connectivity'),
                                                                trainable=False)

            
            
            self.is_initialized = False
            self.is_built = False
            
            

    def build(self):

        self.predictions, self.states = self.forward_pass()
        self.loss = LossFunction(self.params).set_model_loss(self)
        self.reg = Regularizer(self.params).set_model_regularization(self)
        self.reg_loss = self.loss + self.reg
        self.sess = tf.compat.v1.Session()
        self.is_built = True
        return


    def destruct(self):

        if self.is_built:
            self.sess.close()
        tf.compat.v1.reset_default_graph()
        return


    def get_effective_W_rec(self):

        W_rec = self.W_rec * self.rec_connectivity
        if self.dale_ratio:
            W_rec = tf.matmul(tf.abs(W_rec), self.Dale_rec, name = "in_1")
        return W_rec


    def get_effective_W_in(self):

        W_in = self.W_in * self.input_connectivity
        if self.dale_ratio:
            W_in = tf.abs(W_in)
        return W_in
        
        
    def get_effective_W_out(self):

        W_out = self.W_out * self.output_connectivity
        # if self.dale_ratio:
        #     W_out = tf.matmul(tf.abs(W_out), self.Dale_out, name = "in_2")
        return W_out



    def get_weights(self):

        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        weights_dict = dict()

        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = self.name):
            if var.name.endswith(':0') and var.name.startswith(self.name):
                name = var.name[len(self.name)+1:-2]
                weights_dict.update({name: var.eval(session = self.sess)})
        weights_dict.update({'W_rec': self.get_effective_W_rec().eval(session=self.sess)})
        weights_dict.update({'W_in': self.get_effective_W_in().eval(session=self.sess)})
        weights_dict.update({'W_out': self.get_effective_W_out().eval(session=self.sess)})
        weights_dict['dale_ratio'] = self.dale_ratio

        return weights_dict
        
        
    def save(self, save_path):
        weights_dict = self.get_weights()
        np.savez(save_path, **weights_dict)
        return        
        
        
    def train(self, trial_batch_generator, train_params={}):

        if not self.is_built:
            self.build()

        learning_rate = train_params.get('learning_rate', 0.001)
        training_iters = train_params.get('training_iters', 50000)
        loss_epoch = train_params.get('loss_epoch', 10)
        verbosity = train_params.get('verbosity', True)
        save_weights_path = train_params.get('save_weights_path', None)
        save_training_weights_epoch = train_params.get('save_training_weights_epoch', 100)
        training_weights_path = train_params.get('training_weights_path', None)
        optimizer = train_params.get('optimizer', 
                                    tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate))
        clip_grads = train_params.get('clip_grads', True)

        if not isgenerator(trial_batch_generator):
            trial_batch_generator = trial_batch_generator.batch_generator()


        if save_weights_path != None:
            if os.path.dirname(save_weights_path) != "" and not os.path.exists(path.dirname(save_weights_path)):
                os.makedirs(path.dirname(save_weights_path))

        if training_weights_path != None:
            if os.path.dirname(training_weights_path) != "" and not os.path.exists(os.path.dirname(training_weights_path)):
                os.makedirs(path.dirname(training_weights_path))

        grads = optimizer.compute_gradients(self.reg_loss)

        if clip_grads:
            grads = [(tf.clip_by_norm(grad, 1.0), var)
                    if grad is not None else (grad, var)
                    for grad, var in grads]


        optimize = optimizer.apply_gradients(grads)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.is_initialized = True

        epoch = 1
        batch_size = next(trial_batch_generator)[0].shape[0]
        losses = []

        while epoch * batch_size < training_iters:

            if self.opto_affect == "multiplicative":
                batch_x, batch_y, output_mask, opto_mask, opto_timecourse, _ = next(trial_batch_generator)
                feed_dict = {self.x: batch_x, 
                             self.y: batch_y, 
                             self.output_mask: output_mask,
                             self.opto_mask: opto_mask,
                             self.opto_timecourse: opto_timecourse}
            elif self.opto_affect == "additive":
                batch_x, batch_y, output_mask, x_opto, opto_mask, use_opto_mask, _ = next(trial_batch_generator)
                feed_dict = {self.x: batch_x, 
                             self.y: batch_y, 
                             self.output_mask: output_mask,
                             self.x_opto: x_opto,
                             self.opto_mask: opto_mask,
                             self.use_opto_mask: use_opto_mask}
            else:
                batch_x, batch_y, output_mask, _ = next(trial_batch_generator)
                feed_dict = {self.x: batch_x,
                             self.y: batch_y,
                             self.output_mask: output_mask}

            self.sess.run(optimize, feed_dict = feed_dict)

            if epoch % loss_epoch == 0:
                reg_loss = self.sess.run(self.reg_loss, feed_dict = feed_dict)
                losses.append(reg_loss)
                if verbosity:
                    print("Iter " + str(epoch * batch_size) + ", Minibatch Loss= " + \
                         "{:.6f}".format(reg_loss))


            if epoch % save_training_weights_epoch == 0:
                if training_weights_path is not None:
                    self.save(training_weights_path + str(epoch))
                    if verbosity:
                        print("Training weights saved in file: %s" % training_weights_path + str(epoch))


            epoch += 1

        if verbosity:
            print("Optimization finished!")

        if save_weights_path is not None:
            self.save(save_weights_path)

        return losses
                    
        
            

        
     
    
    

class Basic(RNN):
    
    
    def recurrent_timestep(self, rnn_in, state):
        new_state = ((1 - self.alpha) * state) \
                    + self.alpha * (
                        tf.matmul(
                            self.transfer_function(state),
                            self.get_effective_W_rec(),
                            transpose_b=True, name = "1")
                        + tf.matmul(
                            rnn_in,
                            self.get_effective_W_in(),
                            transpose_b = True, name = "2")
                        + self.b_rec)\
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                        * tf.random.normal(tf.shape(input = state), mean = 0.0, stddev = 1.0)        
        return new_state
    
    
    
    def output_timestep(self, state):
        output = tf.matmul(self.transfer_function(state),
                          self.get_effective_W_out(), transpose_b = True, name = "3") \
                    + self.b_out
        return output
    
    
    
    def forward_pass(self):
        rnn_inputs = tf.unstack(self.x, axis = 1)
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            state = self.recurrent_timestep(rnn_input, state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(a = rnn_outputs, perm=[1,0,2]), tf.transpose(a = rnn_states, perm = [1,0,2])
            
            
    def test(self, trial_batch):

        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        outputs, states = self.sess.run([self.predictions, self.states],
                                           feed_dict = {self.x: trial_batch})
    
        return outputs, states
        

        
        
        
        
        

                
class Basic_optomul(RNN):
    
    def __init__(self, params):
        super(Basic_optomul, self).__init__(params)
        
        assert self.opto_affect == "multiplicative"
        
        # type of opto inactivation for a given trial
        self.opto_mask = tf.compat.v1.placeholder("float", [None, self.N_rec, self.N_rec])
        # timecouse of inactivation
        self.opto_timecourse = tf.compat.v1.placeholder("float", [None, self.N_steps])
        # reconstructed weight matrix at any time step formed by multiplying opto_mask with opto_timecouse at timestep t
        self.opto_rec = tf.compat.v1.placeholder("float", [None, self.N_rec, self.N_rec])
    
    
    
    
    def recurrent_timestep(self, rnn_in, ocourse, state):
        
        self.opto_rec = tf.multiply(self.opto_mask * tf.reshape(ocourse, (-1, 1, 1)),
                                    self.get_effective_W_rec(),
                                    name = "opto_mask_1") * 0.9
        
        new_state = ((1-self.alpha) * state) \
            + self.alpha * (tf.squeeze(tf.matmul(tf.expand_dims(self.transfer_function(state),axis = 1),
                                                self.get_effective_W_rec() - self.opto_rec,
                                                transpose_b = True, name = "1"), 1) \
                + tf.matmul(
                    rnn_in,
                    self.get_effective_W_in(),
                    transpose_b=True, name="2")
                + self.b_rec)\
            + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
              * tf.random.normal(tf.shape(input=state), mean=0.0, stddev=1.0)       
        
        return new_state
    
    
    def output_timestep(self, state):
        output = tf.matmul(self.transfer_function(state),
                          self.get_effective_W_out(), transpose_b = True, name = "3") \
                    + self.b_out
        return output
    
    
    
    
    def forward_pass(self):
        
        rnn_inputs = tf.unstack(self.x, axis = 1)
        opto_timecourse = tf.unstack(self.opto_timecourse, axis = 1)
        
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input, ocourse in zip(rnn_inputs, opto_timecourse):
            state = self.recurrent_timestep(rnn_input,
                                            ocourse,
                                            state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(a = rnn_outputs, perm=[1,0,2]), tf.transpose(a = rnn_states, perm = [1,0,2])
            
        
                
    def test(self, trial_batch, opto_mask, opto_timecourse):

        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        outputs, states = self.sess.run([self.predictions, self.states],
                                           feed_dict = {self.x: trial_batch,
                                                        self.opto_mask: opto_mask,
                                                        self.opto_timecourse: opto_timecourse})
      

        return outputs, states
        
        
            
                              
            
            
            
                                            
            
class Basic_optoadd(RNN):
    
    def __init__(self, params):
        
        super(Basic_optoadd, self).__init__(params)
        
        self.num_opto = len(self.params['opto_train'])
        assert self.opto_affect == "additive"
        
        # opto connectivity matrix
        with tf.compat.v1.variable_scope(self.name) as scope:
            self.W_perturb = tf.compat.v1.get_variable('W_perturb', 
                                                       [self.N_rec, self.num_opto],
                                                       initializer =self.initializer.get('W_perturb'),
                                                       trainable = False)
        # timecourse of opto
        self.x_opto = tf.compat.v1.placeholder("float", [None, self.N_steps, self.num_opto]) 
        # projection specific inactivations?
        self.use_opto_mask = tf.compat.v1.placeholder("float", [None, 1])
        # make for projection specfic inactivations
        self.opto_mask = tf.compat.v1.placeholder("float", [None, self.N_rec, self.N_rec])
        # dummy variable
        self.opto_input = tf.compat.v1.placeholder("float", [None, self.N_rec])

    
    
    
    def recurrent_timestep(self, rnn_in, opto_in, state):
        
        self.opto_input =  tf.matmul(opto_in,
                                     self.W_perturb,
                                     transpose_b = True,
                                     name = "opto_1")
    
        new_state = ((1-self.alpha) * state) \
                    + self.alpha * (
                          tf.matmul(self.transfer_function(state),
                                   self.get_effective_W_rec(),
                                    transpose_b = True,
                                    name = "rec")
                        + tf.matmul(rnn_in,
                                    self.get_effective_W_in(),
                                    transpose_b = True,
                                    name = "input_1")
                        + tf.multiply((1-self.use_opto_mask),self.opto_input)
                        + tf.multiply(self.use_opto_mask,
                                      tf.squeeze(tf.matmul(tf.expand_dims(self.transfer_function(state + self.opto_input),
                                                                                                 axis = 1),
                                                                                  tf.multiply(self.opto_mask,
                                                                                              self.get_effective_W_rec(),
                                                                                              name = "opto_mask_1"),
                                                                                  transpose_b = True,
                                                                                  name = "opto_2"),1))
                        - tf.multiply(self.use_opto_mask,
                                      tf.squeeze(tf.matmul(tf.expand_dims(self.transfer_function(state),axis = 1),
                                                                                  tf.multiply(self.opto_mask,
                                                                                              self.get_effective_W_rec(),
                                                                                              name = "opto_mask_2"),
                                                                                  transpose_b = True,
                                                                                  name = "opto_3"),1))
                        + self.b_rec)\
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                        * tf.random.normal(tf.shape(input = state), mean = 0.0, stddev = 1.0)
        
        return new_state
        
        
    
        
    def output_timestep(self, state):
        output = tf.matmul(self.transfer_function(state),
                          self.get_effective_W_out(), transpose_b = True, name = "3") \
                    + self.b_out
        return output
    
    
    
    
    def forward_pass(self):
        
        rnn_inputs = tf.unstack(self.x, axis = 1)      
        opto_inputs = tf.unstack(self.x_opto, axis = 1)
        
        state = self.init_state
        rnn_outputs = []
        rnn_states = []
        for (rnn_input, opto_input) in zip(rnn_inputs, opto_inputs):
            state = self.recurrent_timestep(rnn_input, opto_input, state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(a = rnn_outputs, perm=[1,0,2]), tf.transpose(a = rnn_states, perm = [1,0,2])
                                            
                                            
                                            
                                            
    def test(self, trial_batch, x_opto, opto_mask, use_opto_mask):

        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        outputs, states = self.sess.run([self.predictions, self.states],
                                           feed_dict = {self.x: trial_batch,
                                                        self.x_opto: x_opto,
                                                        self.opto_mask: opto_mask,
                                                        self.use_opto_mask: use_opto_mask})

        return outputs, states
                                            
            
                                            
                                    
        
        
        
            