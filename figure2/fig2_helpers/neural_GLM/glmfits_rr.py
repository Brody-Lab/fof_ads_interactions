from common_imports import *
import sys
sys.path.insert(1, '../../../figure_code/')


from my_imports import *
from copy import deepcopy
from helpers.phys_helpers import get_sortCells_for_rat, datetime, split_trials
from helpers.physdata_preprocessing import load_phys_data_from_Cell

from glm import glm_rr, glmRRDataset
from train_model import train_model, l2_regularizer
from glmfits_utils import *
from basis_kernels import make_kernel


p = dict()
p['ratnames'] = SPEC.RATS
p['regions'] = SPEC.REGIONS
p['cols'] = SPEC.COLS
p['align_to'] = 'cpoke_in'
p['window'] = [0, 2000]
p['fr_thresh'] = 1.0 # firing rate threshold for including neurons
p['binsize'] = 5
p['pre_mask'] = None
p['post_mask'] = None
p['filter_w'] = None
p['filter_type'] = None
p['covariates'] = [
    'leftBups',
    'rightBups',
    'cpoke_in',
    'stereo_click',
    'choiceL',
    'choiceR']

# fitting related
p['num_folds'] = 5
p['num_repeats'] = 10
p['num_bootstraps'] = 10
p['num_epochs'] = 40000
p['regularization_range'] = [1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0., 0.1]
dt = p['binsize']/1000
p['dt'] = dt

# this dictionary specified rank interactions
rank_dict = dict()
rank_dict['rank'] = int(sys.argv[3])
rank_dict['num'] = 15
rank_dict['duration'] = 0.5
rank_dict['peak_range'] = [0.005, 0.4]
rank_dict['log_scaling'] = True
rank_dict['time_offset'] = 0.0
rank_dict['log_offset'] = 0.3
rank_dict['weights'] = np.random.rand(rank_dict['rank'] * rank_dict['num'])
rank_dict['kernels'] = make_kernel(rank_dict, dt)

# FOF to ADS, and then ADS to FOF
reg_order = [[p['regions'][0],p['regions'][1]],
            [p['regions'][1],p['regions'][0]]]



SAVEDIR = SPEC.RESULTDIR + 'neural_rr_GLM/'
summary = dict()
summary['model_dict'] = dict()

# rat = p['ratnames'][0]
rat = p['ratnames'][int(sys.argv[1])]

print("\n\n\n\n===== RAT: {} =====".format(rat))
files, this_datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)
file = files[int(sys.argv[2])]
# file = files[0]

print('\n\nProcessing file: {}'.format(file))
fname =  SAVEDIR + file[:21] + '_neural_GLM_RR_rank' + str(rank_dict['rank']) + datetime()[5:] + '.npy'

df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + file)
df_cell = df_cell[df_cell.stim_fr > 1.0].reset_index()

# then get stimulus 
X = get_X(df_trial, p)

# then make covariate dict
covar_dict = get_covar_dict(p)

for this_ff in range(len(reg_order)):
    
    fit_name = reg_order[this_ff][0] + '_' + reg_order[this_ff][1]

    # first get spikes (cause it is easy!)
    Y = get_Y(df_cell, df_trial, p, region = reg_order[this_ff][1])
    num_Y = Y.shape[2]
    
    # then get activity of the other region
    p_input = deepcopy(p)
    p_input['window'] = [x - p_input['binsize'] for x in p_input['window']]
    Y_input = get_Y(df_cell, df_trial, p_input, region = reg_order[this_ff][0])
    rank_dict['num_inputs'] = Y_input.shape[2]
    
    # make a dataset for fitting
    dataset = glmRRDataset(X, Y_input, Y)
    
    # initialize the glm model
    this_glm = glm_rr(
        covar_dict = covar_dict,
        num_Y = num_Y,
        rank_dict = rank_dict,
        dt = dt)
    
    # train!
    model_dict = train_model(
        this_glm,
        dataset,
        regularizer = l2_regularizer,
        regularization_range = p['regularization_range'],
        num_epochs = p['num_epochs'],
        verbose = True,
        num_folds = p['num_folds'],
        num_repeats = p['num_repeats'],
        num_bootstraps = p['num_bootstraps'])
    
    summary['model_dict'][fit_name] = model_dict
    

summary['filename'] = file
summary['p'] = p
summary['covar_dict'] = covar_dict
summary['rank_dict'] = rank_dict
    
np.save(fname, summary)

    