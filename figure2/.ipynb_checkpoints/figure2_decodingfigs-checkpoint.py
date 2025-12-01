import sys, re
sys.path.insert(1, '../../figure_code/')
from my_imports import *
import pingouin as pg
from helpers.phys_helpers import get_sortCells_for_rat, savethisfig

def get_filenames(decoding_dir):
    
    filenames = sorted([fn for fn in os.listdir(decoding_dir) if re.findall(".npy", fn)])
    param_file = [fn for fn in filenames if 'params' in fn][0]  
    p = np.load(decoding_dir + param_file, allow_pickle = True).item()
    filenames.remove(param_file)
    return p, filenames


def get_file_ID(fn, df_data, reg):
    file_idx = np.where((df_data['file_ID'].str.contains(fn[:18])) & (df_data['region'] == reg))[0][0]
    this_file = df_data['file_ID'].iloc[file_idx]
    return file_idx, this_file


def compute_t(start_time, binsize, tpts):    
    return [1e-3*(start_time + binsize * it + binsize/2) for it in tpts]


def add_evidence_decoding(dir_ev_decoding, filenames, p, df_data):
    
    this_dict = dict()
    this_dict['region'] = []
    this_dict['file_ID'] = []
    this_dict['evidence_dec_accuracy'] = []
    this_dict['evidence_dec_accuracy_t'] = []
    this_dict['evidence_dec_accuracy_left'] = []
    this_dict['evidence_dec_accuracy_right'] = []

    for fn in filenames:
        summary = np.load(dir_ev_decoding + fn, allow_pickle = True).item()
        summary = summary['clickson_masked']

        for reg in p['regions']:
            file_idx, this_file = get_file_ID(fn, df_data, reg)
            this_dict['region'].append(reg)
            this_dict['file_ID'].append(this_file)

            # add delta clicks decoding
            this_dict['evidence_dec_accuracy'].append(summary['all'][reg]['accuracy'])
            t = compute_t(p['start_time'][0], p['binsize'], summary['all'][reg]['tpts'])
            this_dict['evidence_dec_accuracy_t'].append(t) 

            # add for choice controlled analysis
            this_dict['evidence_dec_accuracy_left'].append(summary['left_going'][reg]['accuracy'])     
            this_dict['evidence_dec_accuracy_right'].append(summary['right_going'][reg]['accuracy'])     

    df_data = pd.merge(
        df_data, 
        pd.DataFrame.from_dict(this_dict),
        how='left', 
        left_on=['region','file_ID'], 
        right_on = ['region','file_ID'])

    return df_data



def plot_decoding_accuracy(df, var_acc, reg, ax):
    
    if 'evidence' in var_acc:
        var_t = 'evidence_dec_accuracy_t'
    else:
        var_t = var_acc + '_t'
    ls = '-'
    label = reg
    
    num_files = len(df)
    max_idx = np.argmax([(np.shape(df[var_acc].values[i])) for i in range(num_files)])
    vals = np.nan * np.zeros((num_files, np.shape(df[var_acc].values[max_idx])[0]))   
    for i in range(num_files):
        vals[i,:len(df[var_acc].values[i])] = df[var_acc].values[i]

    npts = np.sum(~np.isnan(vals), axis =0)
    vals_mean = np.nanmean(vals, axis = 0)
    vals_sem = np.nanstd(vals, axis = 0)/np.sqrt(npts)
    
    ax.plot(df[var_t].values[max_idx],
                    vals_mean, 
                    c = SPEC.COLS[reg],
                    ls = ls,
                    label = label)
    ax.fill_between(df[var_t].values[max_idx],
                            vals_mean - vals_sem, 
                            vals_mean + vals_sem,
                            color = SPEC.COLS[reg],
                            alpha = 0.3)
    ax.set_xlabel('Time from clicks on [s]', fontsize = 14)
    ax.set_xticks([0.,0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon = False)
    
    



if __name__ == "__main__":
    
    # define a pandas dataframe to merge decoding information and individal sessions
    allfiles = []
    for rat in SPEC.RATS:
        files, _ = get_sortCells_for_rat(rat, SPEC.DATADIR)
        allfiles.extend(files)
    df_data = pd.DataFrame()
    df_data['file_ID'] = np.tile(np.sort(allfiles), len(SPEC.REGIONS))
    df_data['region'] = np.repeat(SPEC.REGIONS, len(allfiles))
    

    # Evidence decoding, with separate decoding for left and right going trials
    EV_DECODING_DIR = 'evidence_decoding/'
    this_dir = os.path.join(SPEC.RESULTDIR, EV_DECODING_DIR)
    p_ev_decoding, filenames_ev_decoding = get_filenames(this_dir)
    df_data = add_evidence_decoding(
        this_dir, 
        filenames_ev_decoding, 
        p_ev_decoding, 
        df_data)
    
    
    # plot evidence decoding accuracy
    fig = plt.figure(figsize = (3,3))
    ax = plt.gca()
    for r, reg in enumerate(SPEC.REGIONS):
        mask = 'region == @reg'
        df = df_data.query(mask).reset_index()

        plot_decoding_accuracy(df, 'evidence_dec_accuracy', reg, ax)
        ax.set_ylim([0.,0.6])
        ax.set_ylabel('Corr(#R-#L clicks, predicted)', fontsize = 14)
    ax.tick_params(axis='both', length = 4, color= 'k', which='major', labelsize=12, direction = 'out')
    sns.despine()
    savethisfig(SPEC.FIGUREDIR + "figure2/", 'figure2_evidnecedecoding_alltrials')
    
    
    # plot evidence decoding accuracy for left and right going trials
    fig, axs = plt.subplots(2,1, figsize = (2,3.5), sharex = True, sharey = True)
    for r, reg in enumerate(SPEC.REGIONS):
        mask = 'region == @reg'
        df = df_data.query(mask).reset_index()
        
        plot_decoding_accuracy(df, 'evidence_dec_accuracy_left', reg, axs[0])
        axs[0].set_xlabel("")
        axs[0].set_title('LEFT going trials')
        
        plot_decoding_accuracy(df, 'evidence_dec_accuracy_right', reg, axs[1])
        axs[1].set_ylim([0.,0.6])
        axs[1].set_title('RIGHT going trials')
        axs[1].set_xlabel('Time from clicks on [s]', fontsize = 10)
    
    ax.tick_params(axis='both', length = 4, color= 'k', which='major', labelsize=10, direction = 'out')
    
    fig.text(0.0, 0.3, 'Corr(#R-#L clicks, predicted)', rotation='vertical', fontsize = 10)
    sns.despine()
    plt.tight_layout()
    savethisfig(SPEC.FIGUREDIR + "figure2/", 'figure2_evidnecedecoding_LandRtrials')
    
    
    # PRINTING STATISTICS OF DIFFERENCES IN DECODING PERFORMANCE TO A FILE
    output_file = open(SPEC.RESULTDIR + "figure2_evidencedecoding_summary.txt",'w')
    decision_variables = ['all', 'left_going', 'right_going']
    
    for dec_var in decision_variables:
        
        print("\n\nSUMMARY of 2-way repeated measures ANOVA for evidence decoding accuracy for %s trials" % dec_var, file = output_file)
        comp = dict()
        comp['t'] = []
        comp['filename'] = []
        comp['region'] = []
        comp['accuracy'] = []
    
        for fn in filenames_ev_decoding:
            
            summary = np.load(this_dir + fn, allow_pickle=True).item()
            summary = summary['clickson_masked']
            
            for reg in SPEC.REGIONS:
                # this package has stupid missing value handling,
                # so we want to make sure that all files have
                # all time points
                if dec_var == "all":
                    idx = np.arange(16)
                elif dec_var == "right_going":
                    # for right going
                    idx = np.arange(11)
                elif dec_var == "left_going":
                    # for left going
                    idx = np.arange(12)
                    
                comp['accuracy'].append(summary[dec_var][reg]['accuracy'][idx])
                t = compute_t(p_ev_decoding['start_time'][0], p_ev_decoding['binsize'], summary[dec_var][reg]['tpts'])
                comp['t'].append(np.array(t)[idx])
                
                for i in range(len(np.array(t)[idx])):
                    comp['filename'].append(fn)
                    comp['region'].append(reg)
                
        comp['accuracy'] = np.concatenate(comp['accuracy']).ravel()
        comp['t'] = np.concatenate(comp['t']).ravel()
        pd_df = pd.DataFrame(comp)
        
        # Make sure no nans are in the data
        assert np.isnan(pd_df['accuracy']).any() == False
        stats = pg.rm_anova(data = pd_df, dv = 'accuracy', within = ['t', 'region'], subject = 'filename')
        print(stats, file = output_file)


