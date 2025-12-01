import numpy as np

from scipy.stats import expon
from scipy.special import expit


class PoissonClicks(object):
    
    
    def __init__(self, **params):
        
        self.dt = params.get('dt', 10)
        self.tau = params.get('tau', 100)
        self.N_out = params.get('N_out', 1)
        self.N_rec = params.get('N_rec', 300)
        self.N_batch = params.get('N_batch', 64)
        self.T = params.get('T', 1700)
        self.opto_affect = None
        self.N_rec_multiply = int(self.N_rec/300)
        
        # trial selection
        self.gamma_list = params.get('gamma_list', np.linspace(-3.5, 3.5, 8))
        self.p_probe = params.get('p_probe', 0.)
        self.probe_duration = params.get('probe_duration', 1000)

        # accumulator params
        self.acc_bound = params.get('acc_bound', 8.)
        self.acc_sigma_sens = params.get('acc_sigma_sens', 2.)
        self.acc_lambda = params.get('acc_lambda', 0.)
        self.acc_lapse = params.get('acc_lapse', 0)
        self.history_mod = params.get('history_mod', None)
        
        # inferred params
        self.alpha = (1.0 * self.dt) / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))
        
        self.N_in = 2
        if self.acc_lapse != 0:
            self.N_in += 1
        if self.history_mod is not None:
            self.N_in += 1
            
            
            
    def generate_trial_params(self, batch, trial):
        
        prm = dict()
        prm['gamma'] = np.random.choice(self.gamma_list)
        prm['is_probe'] = np.random.rand() < self.p_probe
        prm['is_lapse'] = np.random.rand() <= self.acc_lapse
        
        # opto is always off - declaring it here makes implementation of 
        # other child classes a bit simpler
        prm['is_opto'] = False
        prm['opto_side'] = None
            
        if prm['is_probe'] == 1:
            prm['stim_duration'] = self.probe_duration
            prm['onset_time'] = 1500 - prm['stim_duration']
        else:
            prm['onset_time'] = 500 + 800*np.random.rand()
            prm['stim_duration'] = 1500 - prm['onset_time']
            
        if self.history_mod is not None:
            prm['history_bias'] = np.random.choice([-2,2])
        else:
            prm['history_bias'] = 0.
            
        return prm
    
    
    
    def generate_trial(self, prm):
        
        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask   = np.zeros([self.N_steps, self.N_out])
        a      = np.zeros(self.N_steps)
        
        # generate clicks
        total_rate = 40
        r_rate   = total_rate*np.exp(prm['gamma'])/(1+np.exp(prm['gamma']))
        l_rate   = total_rate - r_rate
        l_clicks = np.cumsum(expon.rvs(scale = 1/l_rate, size = total_rate))*1e3
        r_clicks = np.cumsum(expon.rvs(scale = 1/r_rate, size = total_rate))*1e3
        l_clicks = prm['onset_time'] + l_clicks[l_clicks <  prm['stim_duration']]
        r_clicks = prm['onset_time'] + r_clicks[r_clicks <  prm['stim_duration']]
        prm['left_clicks'] = l_clicks
        prm['right_clicks'] = r_clicks
        prm['Δclicks'] = len(r_clicks) - len(l_clicks)
        
        # add history bias 
        if self.history_mod is not None:
            if self.acc_lapse > 0:
                x_data[:5, 3] = prm['history_bias'] 
            else:
                x_data[:5, 2] = prm['history_bias']
                
            
        flag = 1 # for adding history bias to the first time point of the accumulator
        flag_bound = 0 # to mark is the bound has been hit or not
        
        dd = np.exp(self.dt * self.acc_lambda)
        
        for t in range(self.N_steps):
            
            tix = t*self.dt
            
            # effect of leak
            if tix > prm['onset_time']:
                a[t] = a[t-1]*dd
                if flag == 1:
                    a[t] += prm['history_bias']
                    flag = 0
                    
            # left lick inputs:
            if len(l_clicks) > 0:
                while (l_clicks[0] <= tix):
                    increment = 1+ np.random.randn() * self.acc_sigma_sens
                    if increment < 0:
                        x_data[t,1] += np.abs(increment)
                    else:
                        x_data[t,0] += np.abs(increment)
                    a[t] -= increment
                    l_clicks = l_clicks[1:]
                    if len(l_clicks) == 0:
                        break
                        
            # right click inputs:
            if len(r_clicks) > 0:
                while(r_clicks[0] <= tix):
                    increment = 1 + np.random.randn() * self.acc_sigma_sens
                    if increment < 0:
                        x_data[t, 0] += np.abs(increment)
                    else:
                        x_data[t, 1] += np.abs(increment)
                    a[t] += increment
                    r_clicks = r_clicks[1:]
                    if len(r_clicks) == 0:
                        break
                    
            # if bound has been hit dont update accumulator
            if flag_bound == 1:
                a[t] = a[t-1]
                
            # has the bound been hit?
            if np.abs(a[t]) >= self.acc_bound:
                a[t] = np.sign(a[t])*self.acc_bound
                flag_bound = 1
            
            # set lapse inputs
            if prm['is_lapse'] == True:
                x_data[t,2] += 1
                
            # set choice
            if tix > prm['onset_time']:
                
                if prm['is_lapse'] == True:
                    y_data[t] = 0.
                elif (prm['is_opto'] == True) & (prm['opto_side'] != None):
                    y_data[t] = 1.0 if prm['opto_side'] == 1 else -1.0
                else:
                    y_choice = np.random.rand() <= expit(5*a[t])
                    y_data[t] = 1.0 if y_choice == True else -1.0
                    
                    
            # set mask
            if tix > (prm['onset_time'] + prm['stim_duration']):
                mask[t] = 1
                # freeze target value once stimulus is over
                y_data[t] = y_data[t-1]
                
                
        prm['accumulator'] = a
        if prm['is_lapse'] == True:
            prm['choice_target_end'] = np.random.choice([-1,1])
        elif (prm['is_opto'] == True) & (prm['opto_side'] != None):
            prm['choice_target_end'] = prm['opto_side']
        else:
            prm['choice_target_end'] = np.mean(y_data*mask) > 0
            

        return {'x_data': x_data, 
                'y_data': y_data, 
                'mask': mask, 
                'prm': prm}
      
        
        
    def batch_generator(self):
        
        batch = 1
        while batch > 0:
            
            x_data = []
            y_data = []
            mask   = []
            params = []
            
            for trial in range(self.N_batch):
                p = self.generate_trial_params(batch, trial)
                out_dict = self.generate_trial(p)
                x_data.append(out_dict['x_data'])
                y_data.append(out_dict['y_data'])
                mask.append(out_dict['mask'])
                params.append(out_dict['prm'])
                
            batch += 1
            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(params)
                
                
                
    def get_trial_batch(self):
        
        return next(self.batch_generator())
    
    
    
    
    def get_task_params(self):
        
        return self.__dict__
                
                
        
        
        
        
        
class PClicks_optomul(PoissonClicks):
    
    def __init__(self, **params):
        
        super(PClicks_optomul, self).__init__(**params)
        self.frac_opto = params.get('frac_opto', 0.2)
        self.opto_affect = "multiplicative"

        # opto dict
        opto_dict = {'left_FOF': {'index': np.ix_(range(300), range(40)),
                                  '1_half': None,
                                  '2_half': -1},
                     'right_FOF': {'index': np.ix_(range(300), range(40,80)),
                                   '1_half': None,
                                   '2_half': 1},
                     'left_ADS': {'index': np.ix_(range(300), range(200,250)), 
                                   '1_half': -1, 
                                   '2_half': -1},
                     'right_ADS': {'index': np.ix_(range(300), range(250,300)), 
                                    '1_half': 1, 
                                    '2_half': 1},
                     'left_proj': {'index': np.ix_(range(200,250), range(80)), 
                                   '1_half': -1, 
                                   '2_half': -1},
                     'right_proj': {'index': np.ix_(range(250,300), range(80)), 
                                    '1_half': 1, 
                                    '2_half': 1},
                     'left_ADSproj': {'index': np.ix_([*range(0,40), *range(180,190)], range(80,180)),
                                      '1_half': 0,
                                      '2_half': 0},
                     'right_ADSproj': {'index': np.ix_([*range(40,80), *range(190,200)], range(80,180)),
                                       '1_half': 1,
                                       '2_half': 1}}
                
        self.opto_train = params.get('opto_train', ['left_FOF', 'right_FOF', 'left_ADS', 'right_ADS', 'left_proj', 'right_proj'])
        self.opto_dict = params.get('opto_dict', {k: opto_dict[k] for k in self.opto_train})        
        
    
    
    def generate_trial_params(self, batch, trial):
        
        prm = dict()
        prm['gamma'] = np.random.choice(self.gamma_list)
        prm['is_probe'] = np.random.rand() < self.p_probe
        prm['is_opto'] = np.random.rand() <= self.frac_opto
        
        if prm['is_opto'] == True:
            prm['is_lapse'] = False
            prm['is_probe'] = 1
            prm['opto_grp'] = np.random.choice(self.opto_train)
            prm['opto_time'] = np.random.choice(['1_half', '2_half'])
            prm['opto_side'] = self.opto_dict[prm['opto_grp']][prm['opto_time']]
            prm['opto_weights'] = self.opto_dict[prm['opto_grp']]['index']
        else:
            prm['is_lapse'] = np.random.rand() <= self.acc_lapse
            prm['opto_side'] = None
            prm['opto_grp'] = None
            prm['opto_time'] = None
            prm['opto_weights'] = None
            
        if prm['is_probe'] == 1:
            prm['stim_duration'] = self.probe_duration
            prm['onset_time'] = 1500 - prm['stim_duration']
        else:
            prm['onset_time'] = 500 + 800*np.random.rand()
            prm['stim_duration'] = 1500 - prm['onset_time']
            
        if self.history_mod is not None:
            prm['history_bias'] = np.random.choice([-2,2])
        else:
            prm['history_bias'] = 0.
            
        return prm
    
    

    
    def generate_opto_mask(self, prm):
        
        opto_mask = np.zeros((self.N_rec, self.N_rec))
        opto_timecourse = np.zeros(self.N_steps)
        if prm['is_opto'] == True:
            opto_mask[prm['opto_weights']] = 1
            if prm['opto_time'] == '1_half':
                opto_timecourse[int(500/self.dt):int(1000/self.dt)] = 1
            elif prm['opto_time'] == '2_half':
                opto_timecourse[int(1000/self.dt):int(1500/self.dt)] = 1
                
        return opto_mask, opto_timecourse
    
    
    
    
    def generate_trial(self, prm):
        inp_dict = super(PClicks_optomul, self).generate_trial(prm)
        if self.frac_opto > 0:
            inp_dict['opto_mask'], inp_dict['opto_timecourse'] = self.generate_opto_mask(prm)
        else:
            inp_dict['opto_mask'], inp_dict['opto_timecourse'] = None, None
            
        return inp_dict
    
    
    
    
    def batch_generator(self):
        
        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask   = []
            opto_mask = []
            opto_timecourse = []
            params = []

            for trial in range(self.N_batch):
                p = self.generate_trial_params(batch, trial)
                out_dict = self.generate_trial(p)
                x_data.append(out_dict['x_data'])
                y_data.append(out_dict['y_data'])
                mask.append(out_dict['mask'])
                opto_mask.append(out_dict['opto_mask'])
                opto_timecourse.append(out_dict['opto_timecourse'])
                params.append(out_dict['prm'])

            batch += 1
            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(opto_mask), np.array(opto_timecourse), np.array(params)
         
        
     
    
    
    
    
        
class PClicks_optoadd(PClicks_optomul, PoissonClicks):
    

    def __init__(self, **params):    
        super(PClicks_optoadd, self).__init__(**params)
        self.opto_affect = "additive"
     
    
    
    def generate_trial(self, prm):
        
        inp_dict = super(PClicks_optomul, self).generate_trial(prm)
        
        # make opto input
        inp_dict['x_opto'] = np.zeros([self.N_steps, len(self.opto_train)])
        if prm['is_opto'] == True:
            opto_idx = self.opto_train.index(prm['opto_grp'])
            if prm['opto_time'] == '1_half':
                inp_dict['x_opto'][int(500/self.dt):int(1000/self.dt), opto_idx] = 1 
            elif prm['opto_time'] == '2_half':
                inp_dict['x_opto'][int(1000/self.dt):int(1500/self.dt), opto_idx] = 1 
                
        # For projection specific inactivations, inputs go to the parent population
        # but we compute what their effective input would have been using this mask
        # This mask is None for non-projection specific inactivations
        if prm['opto_grp'] == 'left_proj':
            opto_mask = np.zeros((self.N_rec, self.N_rec))
            opto_mask[np.ix_(range(self.N_rec_multiply*200,
                                   self.N_rec_multiply*250),
                             range(self.N_rec_multiply*80))] = 1
            use_opto_mask = 1
        elif prm['opto_grp'] == 'right_proj':
            opto_mask = np.zeros((self.N_rec, self.N_rec))
            opto_mask[np.ix_(range(self.N_rec_multiply*250,
                                   self.N_rec_multiply*300),
                             range(self.N_rec_multiply*80))] = 1
            use_opto_mask = 1
        else:
            opto_mask = np.zeros((self.N_rec, self.N_rec))  
            use_opto_mask = 0
            
        inp_dict['opto_mask'] = opto_mask
        inp_dict['use_opto_mask'] = use_opto_mask

        return inp_dict
    
    
    
    
    def batch_generator(self):

        # return super(PClicks_optomul, self).batch_generator()
        
        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask   = []
            x_opto = []
            opto_mask = []
            use_opto_mask = []
            params = []

            for trial in range(self.N_batch):
                p = self.generate_trial_params(batch, trial)
                out_dict = self.generate_trial(p)
                x_data.append(out_dict['x_data'])
                y_data.append(out_dict['y_data'])
                mask.append(out_dict['mask'])
                x_opto.append(out_dict['x_opto'])
                opto_mask.append(out_dict['opto_mask'])
                use_opto_mask.append(out_dict['use_opto_mask'])
                params.append(out_dict['prm'])

            batch += 1
            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(x_opto), np.array(opto_mask),  np.array(use_opto_mask)[:,None], np.array(params)
         
        
            
        
        
        
    
        
        
        
            
            
            
           
        
        
            
            
            
        
            
    
        
     