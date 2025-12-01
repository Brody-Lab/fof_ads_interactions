import sys
sys.path.insert(1, '../../figure_code/')
from my_imports import *
from helpers.phys_helpers import savethisfig
from scipy import stats
from fig2_helpers.compute_hanks_tuning_curves import *

filepath = SPEC.RESULTDIR + "hanks_tuning_curves/" + "hanks_tuning_curve_analysis_150_to_500.csv"
df_sd = pd.read_csv(filepath)


# PLOT MEDIAN TUNING CURVES ACROSS POPULATION
fig = plt.figure(figsize = (2,3))
ax = plt.gca()
for reg in SPEC.REGIONS:
    query = "region == '{}'".format(reg)
    ax.hist(df_sd.query(query).tun_slope,
            bins = np.linspace(0,0.9,10),
            alpha = 0.8,
            edgecolor = 'white',
            color = SPEC.COLS[reg],
            label = reg)
    ax.axvline(np.median(df_sd.query(query).tun_slope),
                c = SPEC.COLS[reg],
                linestyle = '--')
    print(reg, np.median(df_sd.query(query).tun_slope))
ax.legend(frameon = False)
ax.set_xlabel('Tuning curve slope')
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel('Number of neurons')
ax.set_ylim([-40,40])
ax.axhline(0, c = 'k')
fig.suptitle(' FOF: ' + str(len(df_sd.query('region == "FOF"' ))) + ' , ADS: ' +  str(len(df_sd.query('region == "ADS"' ))) )
sns.despine()
savethisfig(SPEC.FIGUREDIR + "figure2/",'tuning_curve_slopes_FOF_ADS')



# PRINT STATISTICS
output_file = open(SPEC.RESULTDIR + "figure2_hankstuning_summary.txt",'w')
query_FOF = 'region == "FOF"' 
query_ADS = 'region == "ADS"'
print('FOF neurons: ' + str(len(df_sd.query('region == "FOF"' ))) + ' , ADS neurons: ' +  str(len(df_sd.query('region == "ADS"' ))),
    file = output_file)

print("ADS (median + std) = " + str(np.median(df_sd.query(query_ADS)['tun_slope'])), 
        str(np.std(df_sd.query(query_ADS)['tun_slope'])),
        file = output_file)
print("FOF (median + std) = " + str(np.median(df_sd.query(query_FOF)['tun_slope'])),
        str(np.std(df_sd.query(query_FOF)['tun_slope'])),
        file = output_file)

_, pval = stats.mannwhitneyu(
    np.array(df_sd.query(query_ADS)['tun_slope'], dtype = "float"), 
    np.array(df_sd.query(query_FOF)['tun_slope'], dtype = "float"))
print("Mann-whitney U test, pval = " + str(pval), file = output_file)




## SUPPLEMENTARY PLOT FOR INDIVIDUAL RATS
# fig, axs = plt.subplots(5,1,constrained_layout=True, sharex=True, sharey=True, figsize=(1.5, 7))
# for ax, rat in zip(axs.ravel(), SPEC.RATS):
#     for reg in SPEC.REGIONS:
#         query = 'region == @reg and rat == @rat'
#         ax.hist(df_sd.query(query).tun_slope, 
#                 bins = np.linspace(0,1,40),
#                 alpha = 0.6, 
#                 color = SPEC.COLS[reg], 
#                 label = reg)
#         ax.axvline(
#             np.median(np.abs(df_sd.query(query).tun_slope)), 
#             c = SPEC.COLS[reg])
#     ax.set_title(rat)
# axs[4].set_xlabel('Tuning curve slope')
# fig.text(-0.1, 0.5, "Number of neurons", rotation=90, va="center")
# sns.despine()
# plt.savefig(SPEC.FIGUREDIR + "figure2/", "SUPPfigure2_hanks_tuning_summary_histograms")



# EXAMPLE TUNING CURVES


cell_IDs = ["X062_20200320_255", # ADS categorical
            "X062_20200319_250", # ADS graded
            "X046_20190923_23",  # ADS graded
            "X062_20200320_209", # FOF categorical
            "X062_20200319_177"  # FOF graded
           ]

# cell_IDs = ["X062_20200320_255"] # ADS categorical

p = dict()

p['fr_thresh'] = 1.0
p['side_pref_thresh'] = 0.05
p['auc_thresh'] = 0.05
p['correct_only'] = False

p['cols'] = SPEC.COLS

p['align_to'] = "model_start"
p['post_mask'] = "clicks_off"
p['window'] = [0, 1500]
p['binsize'] = 1
p['filter_type'] = "gaussian"
p['filter_w'] = 75

p['neural_delay'] = dict()
p['neural_delay']["FOF"] = 100 # in ms
p['neural_delay']["ADS"] = 100 # in ms

p['t0_ms'] = 150
p['t1_ms'] = 500
savepath = SPEC.RESULTDIR + "hanks_tuning_curves/"


pdf = matplotlib.backends.backend_pdf.PdfPages(savepath + "_hanks_tuning_example_neurons.pdf")

for cell in cell_IDs:
    
    ratname = cell[:4]
    files, this_datadir = get_sortCells_for_rat(ratname, SPEC.DATADIR)
    
    try:
        filename = cell[:9] + '_' + cell[9:11] + '_' + cell[11:13]
        this_file = files[np.where([(f.find(filename) == 0) for f in files])[0][0]]
    except:
        filename = cell[:9] + '-' + cell[9:11] + '-' + cell[11:13]
        this_file = files[np.where([(f.find(filename) == 0) for f in files])[0][0]]
    
    print(cell, ratname, filename, this_file)
    
    df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + this_file)
    df_trial['model_start'] = df_trial['clicks_on'] - (df_trial['stim_dur_s_theoretical'] - df_trial['stim_dur_s_actual'])

    cellnum = np.where(df_cell['cell_ID'] == cell)[0][0]
    print(df_cell.loc[cellnum, 'cell_ID'])
    PSTH_mean = make_psth(df_cell.loc[cellnum, 'spiketime_s'], 
                            df_trial,
                            split_by = None,
                            align_to = "clicks_on", 
                            post_mask = None,
                            window = [-50,50],
                            filter_type = p['filter_type'],
                            filter_w = p['filter_w'],
                            binsize = 1,
                            plot = False)['data'][0]['mean']
    df_cell.loc[cellnum, 'fr_stim_onset'] = np.mean(PSTH_mean) 
    

    # load and rebin backward pass:
    # accum is len(xc)xt (in ms) dim and xc are the bin centers 
    acc_raw, xc_raw = load_backward_pass(this_file, plot = False)
    if p['correct_only'] == True:
        is_hit = np.where(df_trial.is_hit == 1)[0]
        df_trial = df_trial.loc[is_hit].reset_index(drop = True)
        acc_raw = acc_raw[is_hit]
    accum, xc = coarsen_acc_and_xc(acc_raw, xc_raw, df_trial, plotting = False)
    assert len(accum) == len(df_trial)

    n = len(xc)
    nT = [np.shape(accum[tr])[1] for tr in range(len(accum))]
    reg = df_cell.loc[cellnum, 'region']
    
    delay = p['neural_delay'][reg]
    fr = get_neural_data(df_trial, df_cell, cellnum, p, delay, plot = False)

    joint_t, fr_centers = make_joint_distribution(accum, fr)
    fr_at  = compute_fr_given_at(joint_t, fr_centers, nT, n)

    if df_cell.loc[cellnum, 'side_pref'] == 0:
        fr_at = np.flip(fr_at, axis = 1)


    title =  'Cell number: ' + df_cell.loc[cellnum, 'cell_ID'] \
    + ' (' + reg + ')' + ' auc: ' + str(np.round(df_cell.loc[cellnum, 'auc'],2))
    mean_tun, std_tun, popt, popt_rank1, var_exp, fr_mod_range = compute_tuning_curve(fr_at, 
                                                                       fr, 
                                                                       df_trial, 
                                                                       p, 
                                                                       xc, 
                                                                       pdf, 
                                                                       title = title) 
    
    fig, axs = plt.subplots(1,2, figsize = (4,2))
    t0 = int(p['t0_ms']/p['binsize'])
    t1 = int(p['t1_ms']/p['binsize'])    
    fr = fr[:,t0:t1]
     
    n = np.shape(fr_at)[1]

    plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,n)))
    colors = plt.cm.Spectral(np.linspace(0,1,n))[:,:3]
    window = adjust_window(p['window'], p['binsize'])
    edges = np.arange(window[0], window[1] + p['binsize'], p['binsize'])*0.001 # convert to s 
    Δ_fr_at = fr_at[t0:t1,:] 
    
    for i in range(1,n-1):
        axs[0].plot(edges[t0:t1][::30], fr_at[t0:t1,i][::30], color = colors[i], lw = 1.0); 
        axs[0].scatter(edges[t0:t1][::30], fr_at[t0:t1,i][::30], color = colors[i], s = 4); 
    axs[0].set_xlabel('Time from stimulus onset \n - neural response lag [s]')
    axs[0].set_ylabel('Normalized firing rate $\it{r}$')
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[0].tick_params(axis='both', which='minor', labelsize=8)
    
    axs[1].scatter(xc, mean_tun, s = 10, color = colors, zorder = 3)
    axs[1].errorbar(xc, mean_tun, 
                      yerr= std_tun, 
                      ls = "none", 
                      c = 'k', 
                      lw = 0.5, 
                      zorder = 1)
    axs[1].plot(xc[1:-1], sigmoid_4(xc[1:-1], *popt), c = 'k', ls = '-', lw = 0.5, zorder = 2)
    axs[1].set_xlabel('Accumulator value')
    axs[1].set_ylabel('Fractional change in \nfiring rate $\it{r}$')
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].tick_params(axis='both', which='minor', labelsize=8)

    plt.tight_layout()
    sns.despine()
    fig.suptitle(cell + reg + '  Slope = '  + str(np.round(popt[1]*popt[3]/4,2)), fontsize = 5)
    savethisfig(SPEC.FIGUREDIR + "figure2/", cell +'_hankstuningexample')
pdf.close()

