import sys
sys.path.insert(1, '../../figure_code/')
from my_imports import *
import colorsys
from helpers.rasters_and_psths import *
from helpers.phys_helpers import get_sortCells_for_rat, savethisfig
from helpers.physdata_preprocessing import load_phys_data_from_Cell

print("\n\nGenerating plots for Figure 1 (average population responses)\n\n")



# print the population PSTHs from the choice-selective neurons in the two regions
window = [-50, 800]
align_to = 'clicks_on'
filter_w = 50
auc_thresh = 0.05
side_pref_p = 0.05
stim_fr_thresh = 1
binsize = 1
post_mask = "cpoke_out"
filter_type = "half_gaussian"
correct_only = False

pref = dict()
fr_mean = dict()
for reg in SPEC.REGIONS:
    pref[reg] = {i : [] for i in [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]}
    fr_mean[reg] = []
    
for reg in SPEC.REGIONS:
    print('\nProcessing region: {}'.format(reg))
    
    for rat in SPEC.RATS:
        print('Processing rat: {}'.format(rat))
        files, this_datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)
        
        for file in files:
            print('\tProcessing file: {}'.format(file))
            df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + file)
            if correct_only == True:
                df_trial = df_trial[df_trial['is_hit'] == correct_only].reset_index()
            df_cell = df_cell[df_cell['stim_fr'] >= stim_fr_thresh]
            df_cell = df_cell[np.abs(df_cell['auc']-0.5)> auc_thresh]
            df_cell = df_cell[df_cell['side_pref_p'] < 0.05].reset_index()
            df_cell = df_cell[df_cell['region'] == reg].reset_index(drop=True)

            cell_split = split_trials(df_cell, 'side_pref')
            evidence_bins = np.unique(df_trial['gamma'])
            
            for sp in list(cell_split):
                for cix in range(len(cell_split[sp])):
                    c = cell_split[sp][cix]
                    psths = make_psth(
                        df_cell.iloc[c]['spiketime_s'],
                        df_trial,
                        align_to = align_to,
                        window = window,
                        binsize = binsize,
                        split_by = 'gamma',
                        filter_type = filter_type,
                        post_mask = post_mask,
                        filter_w = filter_w,
                        plot = False)['data']                                        
                    psths_mean = make_psth(
                        df_cell.iloc[c]['spiketime_s'],
                        df_trial,
                        align_to = align_to,
                        window = window,
                        binsize = binsize,
                        split_by = None,
                        post_mask = post_mask,
                        filter_type = filter_type,
                        filter_w = filter_w,
                        plot = False)['data'][0]['mean']
                    
                    # flig psths so that preferred (higher firing rate) sides are aligned
                    for i in evidence_bins:
                        if sp == 1:
                            pref[reg][-i].append(psths[i]['mean'].T/np.mean(psths_mean[:60]))
                        else:
                            pref[reg][i].append(psths[i]['mean'].T/np.mean(psths_mean[:60]))
                            
                    fr_mean[reg].append(np.mean(psths_mean[:60]))
                    
                    
                    
# Plotting
t = np.arange(window[0], window[1])
fig, axs = plt.subplots(2,1, figsize = (2.3,4), sharey = False, sharex = True)
for ax, reg in zip(axs, SPEC.REGIONS):
    # palette = iter(sns.color_palette("vlag", 8, desat = 1.))
    palette = iter(sns.diverging_palette(43, 144, s=90,l=50, sep = 1, n = 8))

    for i in evidence_bins:
        color = next(palette)
        ax.plot(t, np.mean(pref[reg][i],  axis = 0), c = color, label = i)
    ax.axvline(0, c = 'k', lw = 1, ls = '--')
    ax.set_xticks(
        np.linspace(0,window[1],5),
        labels=np.linspace(0, window[1], 5).astype(int),
        fontsize = 10,
        fontname = 'helvetica')
    ax.text(50, 0.9*ax.get_ylim()[1], reg, color = SPEC.COLS[reg], fontsize = 14)
    ax.text(
        -280,
        1.0 - 0.125*np.diff(ax.get_ylim()), 
        '[' + str(np.mean(fr_mean[reg]).astype(int)) + 'Hz]', 
        color = 'grey', 
        fontsize = 8)
    
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, 
    labels = ['']*8, 
    handlelength = 0.8, 
    bbox_to_anchor=(1.1, 0.75), 
    fontsize = 8, 
    frameon = False)
fig.text(1.05, 0.7, 'easy', fontsize = 12)
fig.text(1.15, 0.64, 'pref', color = 'grey', fontsize = 12)
fig.text(1.05, 0.565, 'hard', fontsize = 12)
fig.text(1.15, 0.5, 'null', color = 'grey', fontsize = 12)
fig.text(1.05, 0.43, 'easy', fontsize = 12)

axs[1].set_xlabel('Time from stimulus onset [ms]', fontsize = 12)
fig.text(-0.025, 0.35, 'Normalized firing rate', rotation='vertical', fontsize = 12)

sns.despine()
plt.tight_layout()

# savethisfig(SPEC.FIGUREDIR + 'figure1/', 'population_psth_FOF_ADS')


# PRINTING SOME SUMMARY OF THE DATA TO A FILE
output_file = open(SPEC.RESULTDIR + "neuraldata_summary.txt", 'w')

print("Summary statistics of the neural data:", file=output_file)

df_mega_all = []  # Collect unfiltered neurons
for r, rat in enumerate(SPEC.RATS):
    print(rat)
    files, this_datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)

    for f, file in enumerate(files):
        print(file)
        df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + file)
        df_cell['filename'] = file
        df_cell['rat'] = rat
        df_mega_all.append(df_cell)

# Combine all into a single DataFrame
df_mega_all = pd.concat(df_mega_all).reset_index(drop=True)
df_mega_all['auc_thresh'] = np.abs(df_mega_all['auc'] - 0.5)

# Save unfiltered version
df_mega_all.to_pickle(os.path.join(SPEC.RESULTDIR, "df_mega_all.pkl"))

# --- FILTERING ---
df_mega = df_mega_all[df_mega_all['stim_fr'] > 1].reset_index(drop=True)
df_mega.to_pickle(os.path.join(SPEC.RESULTDIR, "df_mega_fr_gt1.pkl"))

# --- Print summary from filtered data ---
print('Number of rats: {}'.format(len(SPEC.RATS)), file=output_file)

print("\nNumber of neurons from each region (stim_fr > 1):", file=output_file)
print(df_mega['region'].value_counts(), file=output_file)

print("\nNumber of neurons with firing rate > 1 and auc_thresh > 0.05:", file=output_file)
print(df_mega.query('auc_thresh > 0.05')['region'].value_counts(), file=output_file)

print("\nNumber of neurons from each region for each session:", file=output_file)
simul_FOF = []
simul_ADS = []
session_files = sorted(df_mega['filename'].unique())

for file in session_files:
    simul_FOF.append(len(df_mega[(df_mega['region'] == "FOF") & (df_mega['filename'] == file)]))
    simul_ADS.append(len(df_mega[(df_mega['region'] == "ADS") & (df_mega['filename'] == file)]))

print("FOF:", simul_FOF, "Mean number:", np.mean(simul_FOF), file=output_file)
print("ADS:", simul_ADS, "Mean number:", np.mean(simul_ADS), file=output_file)

output_file.close()

