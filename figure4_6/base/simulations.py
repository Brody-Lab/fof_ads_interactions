import numpy as np



class BasicSimulator():
    
    
    def __init__(self, rnn_model = None, params = None, weights_path = None, weights = None, transfer_function = None):
        
        
        mapping = {'relu' : self.relu,
                   'gelu' : self.gelu,
                   'gelu_0_2': self.gelu_0_2}

        if transfer_function is not None:
            self.transfer_function = mapping[transfer_function]
            
            
        
        if rnn_model is not None:
            
            self.alpha = rnn_model.alpha
            self.rec_noise = rnn_model.rec_noise
            
            if transfer_function is None:
                self.transfer_function = mapping[rnn_model.transfer_function.__name__]
            
            if rnn_model.transfer_function.__name__ != self.transfer_function.__name__:
                raise UserWarning("Transfer functions are not matched: passed transfer function (" + transfer_function + ") is being used")
                
            if params is not None:
                raise UserWarning("params was passed but will not be used. rnn_model takes precedence")
                
        else:
            self.rec_noise = params.get('rec_noise', 0)
            if params.get('alpha') is not None:
                self.alpha = params['alpha']
            else:
                dt = params['dt']
                tau = params['tau']
                self.alpha = params.get('alpha', (1.0*dt) / tau)
                
                
        self.weights = weights
        if weights is not None:
            if weights_path is not None or rnn_model is not None:
                raise UserWarning("Weights and either rnn_model or weights_path were passed in. Weights from rnn_model and weights_path will be ignored.")
        elif weights_path is not None:
            if rnn_model is not None:
                raise UserWarning("rnn_model and weights_path were both passed in. Weights from rnn_model will be ignored.")
            self.weights = np.load(weights_path)
        elif rnn_model is not None:
            self.weights = rnn_model.get_weights()
        else:
            raise UserWarning("Either weights, rnn_model, or weights_path must be passed in.")
            
        self.W_in = self.weights['W_in']
        self.W_rec = self.weights['W_rec']
        self.W_out = self.weights['W_out']
        
        self.b_rec = self.weights['b_rec']
        self.b_out = self.weights['b_out']
        
        self.init_state = self.weights['init_state']
        
        
    
    def relu(self, x):
        return np.maximum(x, 0)

    def gelu(self, x):
        return (0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x))))
    
    def gelu_0_2(self, x):
        return 0.2 + (0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x))))
    
        
    def rnn_step(self, state, rnn_in, t_connectivity):
        
        new_state = ((1-self.alpha) * state) \
                    + self.alpha * (
                        np.matmul(
                            self.transfer_function(state),
                            np.transpose(self.W_rec * t_connectivity))
                        + np.matmul(rnn_in, np.transpose(self.W_in))
                        + self.b_rec)\
                    + np.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) * \
                      np.random.normal(loc=0.0, scale=1.0, size=state.shape)

        new_output = np.matmul(
                        self.transfer_function(new_state),
                        np.transpose(self.W_out)) + self.b_out

        return new_output, new_state
    
    
    
    
    def run_trials(self, trial_input, t_connectivity=None):
        
        batch_size = trial_input.shape[0]
        rnn_inputs = np.squeeze(np.split(trial_input, trial_input.shape[1], axis=1))
        state = np.expand_dims(self.init_state[0, :], 0)
        state = np.repeat(state, batch_size, 0)
        rnn_outputs = []
        rnn_states = []
        for i, rnn_input in enumerate(rnn_inputs):

            if t_connectivity is not None:
                output, state = self.rnn_step(state, rnn_input, t_connectivity[i])
            else:
                output, state = self.rnn_step(state, rnn_input, np.ones_like(self.W_rec))

            rnn_outputs.append(output)
            rnn_states.append(state)

        return np.swapaxes(np.array(rnn_outputs), 0, 1), np.swapaxes(np.array(rnn_states), 0, 1)



    
    
class BasicSimulator_add(BasicSimulator):
    
    def __init__(self, **kwargs):
        super(BasicSimulator_add, self).__init__(**kwargs)
        
        
    def rnn_step(self, state, rnn_in, opto_in, opto_mask, use_opto_mask):
        
        if use_opto_mask == 1:
            x_opto = np.matmul(self.transfer_function(state + opto_in),
                               np.transpose(self.W_rec * opto_mask)) \
                    - np.matmul(self.transfer_function(state),
                                np.transpose(self.W_rec * opto_mask))
        elif use_opto_mask == 0:
            x_opto = opto_in
            
        elif use_opto_mask == None:
            x_opto = np.zeros(np.shape(state)[0])
        
        new_state = ((1-self.alpha) * state) \
                    + self.alpha * (
                        np.matmul(
                            self.transfer_function(state),
                            np.transpose(self.W_rec))
                        + np.matmul(rnn_in, np.transpose(self.W_in))
                        + x_opto
                        + self.b_rec)\
                    + np.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) * \
                      np.random.normal(loc=0.0, scale=1.0, size=state.shape)

        new_output = np.matmul(
                        self.transfer_function(new_state),
                        np.transpose(self.W_out)) + self.b_out

        return new_output, new_state
        
        
        
        
    def run_trials(self, trial_input, opto_inputs = None, opto_mask = None, use_opto_mask = None):
        
        batch_size = trial_input.shape[0]
        rnn_inputs = np.squeeze(np.split(trial_input, trial_input.shape[1], axis = 1))
        state = np.expand_dims(self.init_state[0,:], 0)
        state = np.repeat(state, batch_size, 0)
        rnn_outputs = []
        rnn_states = []
        
        for i, rnn_input in enumerate(rnn_inputs):
            
            if use_opto_mask is not None:
                output, state = self.rnn_step(state, rnn_input, opto_inputs[i], opto_mask, use_opto_mask)
            else:
                output, state = self.rnn_step(state, rnn_input, opto_inputs, opto_mask, use_opto_mask)
                
            rnn_outputs.append(output)
            rnn_states.append(state)
            
        return np.swapaxes(np.array(rnn_outputs), 0, 1), np.swapaxes(np.array(rnn_states), 0, 1)