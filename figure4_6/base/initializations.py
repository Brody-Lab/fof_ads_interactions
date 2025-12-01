import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class WeightInitializer():
    
    def __init__(self, **params):
        
        self.load_weights_path = params.get('load_weights_path', None)
        N_in = self.N_in = params.get('N_in')
        N_rec = self.N_rec = params.get('N_rec')
        N_out = self.N_out = params.get('N_out')
        self.autapses = params.get('autapses', True)
        self.N_rec_multiply = params.get('N_rec_multiply')
        
        # extract opto info for making W_perturb
        self.opto_affect = params.get('opto_affect')
        self.opto_train = params.get('opto_train') if self.opto_affect is not None else None
        
        
        self.initializations = dict()
        
        if self.load_weights_path is not None:
            
            self.initializations = dict(np.load(self.load_weights_path, allow_pickle = True))
        
        else:
     
            # Connectivity constraints
            self.initializations['input_connectivity'] = params.get('input_connectivity',np.ones((N_rec, N_in)))
            assert(self.initializations['input_connectivity'].shape == (N_rec, N_in))
            self.initializations['rec_connectivity'] = params.get('rec_connectivity',np.ones((N_rec, N_rec)))
            assert(self.initializations['rec_connectivity'].shape == (N_rec, N_rec))
            self.initializations['output_connectivity'] = params.get('output_connectivity', np.ones((N_out, N_rec)))
            assert(self.initializations['output_connectivity'].shape == (N_out, N_rec))         
            
            
            # autapses contrains
            if not self.autapses:
                self.initializations['rec_connectivity'][np.eye(N_rec) == 1] = 0
                
            # Dale's law
            self.initializations['dale_ratio'] = dale_ratio = params.get('dale_ratio', None)
            if type(self.initializations['dale_ratio']) == np.ndarray:
                self.initializations['dale_ratio'] = dale_ratio = self.initializations['dale_ratio'].item()
            if dale_ratio is not None and (dale_ratio <0 or dale_ratio > 1):
                print("Need 0 <= dale_ratio <= 1. dale_ratio was: " + str(dale_ratio))
                raise
                
            dale_vec = np.ones(N_rec)
            if dale_ratio is not None:
                dale_vec[int(dale_ratio * N_rec):] = -1
                dale_rec = np.diag(dale_vec)
                dale_vec[int(dale_ratio * N_rec):] = 1
                dale_out = np.diag(dale_vec)
            else:
                dale_rec = np.diag(dale_vec)
                dale_out = np.diag(dale_vec)
            
            self.initializations['Dale_rec'] = params.get('Dale_rec', dale_rec)
            assert(self.initializations['Dale_rec'].shape == (N_rec, N_rec))
            self.initializations['Dale_out'] = params.get('Dale_out', dale_out)
            assert(self.initializations['Dale_out'].shape == (N_rec, N_rec))
                
            # ----------------------------------
            # Default initializations / optional loading from params
            # ----------------------------------

            self.initializations['W_in'] = params.get('W_in', self.glorot_gauss(self.initializations['input_connectivity']))
            assert(self.initializations['W_in'].shape == (N_rec, N_in))
            self.initializations['W_out'] = params.get('W_out', self.glorot_gauss(self.initializations['output_connectivity']))
            assert(self.initializations['W_out'].shape == (N_out, N_rec))
            self.initializations['W_rec'] = params.get('W_rec', self.glorot_gauss(self.initializations['rec_connectivity']))
            assert(self.initializations['W_rec'].shape == (N_rec, N_rec))

            self.initializations['b_rec'] = params.get('b_rec',np.zeros(N_rec))
            assert(self.initializations['b_rec'].shape == (N_rec,))
            self.initializations['b_out'] = params.get('b_out',np.zeros(N_out))
            assert(self.initializations['b_out'].shape == (N_out,))

            self.initializations['init_state'] = params.get('init_state', .1 + .01 * np.random.randn(N_rec))
            assert(self.initializations['init_state'].size == N_rec)    
          
            if self.opto_affect == "additive":
                self.initializations['W_perturb'] = params.get('W_perturb', self.make_perturbation_weights(self.opto_train, self.N_rec))
        
            
            
    def glorot_gauss(self, connectivity):
        """ Initialize ndarray of shape :data:`connectivity` with values from a glorot normal distribution.

        Draws samples from a normal distribution centered on 0 with `stddev
        = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units and `fan_out` is the number of output units. Respects the :data:`connectivity` matrix.

        Arguments:
            connectivity (ndarray): 1 where connected, 0 where unconnected.

        Returns:
            ndarray(dtype=float, shape=connectivity.shape)

        """
            
        init = np.zeros(connectivity.shape)
        fan_in = np.sum(connectivity, axis = 1)
        init += np.tile(fan_in, (connectivity.shape[1],1)).T
        fan_out = np.sum(connectivity, axis = 0)
        init += np.tile(fan_out, (connectivity.shape[0],1))
        
        return np.random.normal(0, np.sqrt(2/init))
    
    
    def make_perturbation_weights(self, opto_train, N_rec):
        
        # assert N_rec == 300
        k = self.N_rec_multiply
        assert self.initializations['dale_ratio'] == 0.5
        W_perturb = np.zeros((N_rec, len(opto_train)))
        perturb_grp = {'left_FOF': range(k*40),
                       'right_FOF': range(k*40, k*80),
                       'left_ADS': range(k*200, k*250),
                       'right_ADS': range(k*250, k*300),
                       'left_proj': range(k*80),
                       'right_proj': range(k*80)}
                                                                                                    
        for i in opto_train:
            W_perturb[perturb_grp[i], opto_train.index(i)] = -0.25
            
        return W_perturb
        
        
    
    
    def get(self, tensor_name):
        return tf.compat.v1.constant_initializer(self.initializations[tensor_name])
    
    
    def save(self, save_path):
        np.savez(save_path, **self.initializations)
        return
    
    def get_dale_ratio(self):
        return self.initializations['dale_ratio']
    
    
    def balance_dale_ratio(self):
        
        dale_ratio = self.get_dale_ratio()
        if dale_ratio is not None:
            dale_vec = np.ones(self.N_rec)
            dale_vec[int(dale_ratio * self.N_rec):] = dale_ratio/(1-dale_ratio)
            dale_rec = np.diag(dale_vec) / np.linalg.norm(np.matmul(self.initializations['rec_connectivity'],
                                                                   np.diag(dale_vec)),
                                                          axis = 1)[:,np.newaxis]
            self.initializations['W_rec'] = np.matmul(self.initializations['W_rec'], dale_rec)
        return
    
    
    
class GaussianSpectralRadius(WeightInitializer):
    
    def __init__(self, **params):
        
        super(GaussianSpectralRadius, self).__init__(**params)
        
        self.spec_rad = params.get('spec_rad', 1.3)
        self.initializations['W_rec']  =  params.get('W_rec', None)
        
        if self.initializations['W_rec'] is None:
            self.initializations['W_rec']  = np.random.randn(self.N_rec, self.N_rec)
            if self.get_dale_ratio() > 0:
                self.balance_dale_ratio()
            self.initializations['W_rec'] = self.initializations['W_rec'] * self.spec_rad / np.max(abs(np.linalg.eigvals(self.initializations['W_rec'])))
            
   
            
        
        return
            
    
    
