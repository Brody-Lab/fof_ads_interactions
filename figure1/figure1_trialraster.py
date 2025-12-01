import sys
sys.path.insert(1, '../../figure_code/')
from my_imports import *
import colorsys
from helpers.rasters_and_psths import *
from helpers.phys_helpers import get_sortCells_for_rat, savethisfig
from helpers.physdata_preprocessing import load_phys_data_from_Cell

print("\n\nGenerating plots for Figure 1 (single trial raster\n\n")

# print population PSTH from one left/right trial from one session
ratname = "X087"
trial_nums = [133, 164]
files, this_datadir = get_sortCells_for_rat(ratname, SPEC.DATADIR)
df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + files[0])

# threshold firing rate
df_cell = df_cell[df_cell['stim_fr'] >= 0.5]

# remove cells putatively in the corpus callosum
df_cell = df_cell.drop(df_cell.index[np.where((df_cell['DV']> 2) & (df_cell['DV'] <2.5))]).reset_index()

# settings
align_to = 'cpoke_out'
window = {"cpoke_in": [-500, 1000], "cpoke_out": [-2000, 1000]}
binsize = 1
split_by = None
pre_mask = None
post_mask = None

# Figure and color settings
fig, axs = plt.subplots(1,2, figsize = (4,3))
offset = 100*df_cell['DV'][0] + np.max(100*df_cell['DV']) - 100*(df_cell['DV'])
colors = []
for c in range(len(df_cell)):
    cl = SPEC.COLS[df_cell['region'][c]]
    (h,l,s) = colorsys.rgb_to_hls(cl[0], cl[1], cl[2])
    l = 0.2 + 0.8*np.random.rand()
    (r,g,b) = colorsys.hls_to_rgb(h,l,s)
    colors.append([r,g,b])

# collect spikes from each neuron for the given trial number 
for t, trial_num in enumerate(trial_nums):
    raster = []
    for c in range(len(df_cell)):
        this_raster = make_raster(
            df_cell['spiketime_s'][c],
            df_trial,
            plot=False,
            window=window[align_to],
            binsize=binsize,
            pre_mask=pre_mask,
            post_mask=post_mask,
            align_to=align_to,
            split_by=split_by)
        spike_times = [i for i, x in enumerate(this_raster['data'][0][trial_num]) if x!=0]
        raster.append(spike_times)
        
    # format plot
    axs[t].eventplot(
        raster,
        colors=colors,
        linewidths=0.5,
        linelengths=4,
        orientation="horizontal",
        lineoffsets=offset)
    axs[t].axvline(0 - window[align_to][0], c = 'k', ls = '--', lw = 0.5)
    axs[t].axhline(
        offset.values[0]+4, 
        c = 'grey', 
        ls = ':', 
        lw = 0.5, 
        label = str(round(offset.values[0]/100,1)) + "mm")
    axs[t].axhline(
        offset.values[-1]-4, 
        c = 'grey', 
        ls = ':', 
        lw = 0.5, 
        label = str(round(offset.values[-1]/100,1)) + "mm")   
    axs[t].set_xticks(
        np.linspace(0, np.diff(window[align_to])[0], 4),
        labels=np.linspace(window[align_to][0], window[align_to][1], 4).astype(int),
        fontsize = 6,
        fontname = 'helvetica')
    axs[t].spines['left'].set_color('white')
    axs[t].tick_params(axis='y', colors='white')
    sns.despine()
    
axs[1].text(3000, 
            offset.values[-1]-10, 
            str(round(offset.values[0]/100,1)) + "mm", 
            color = 'grey',
            fontsize =7)    
axs[1].text(3000, 
            offset.values[0]-2, 
            str(round(offset.values[-1]/100,1)) + "mm", 
            color = 'grey',
            fontsize = 7)
fig.text(0.85, -0.01, "Depth below \nthe surface", color = 'grey', fontsize = 7)
fig.text(0.02, 0.7, 'FOF', rotation='vertical', color = SPEC.COLS['FOF'])
fig.text(0.02, 0.27, 'ADS', rotation='vertical', color = SPEC.COLS['ADS'])
fig.text(0.06, 0.63, '(' + str(df_cell['region'].value_counts().FOF) + ' neurons)', 
        rotation='vertical',color = SPEC.COLS['FOF'],
        fontsize = 8)
fig.text(0.06, 0.21, '(' + str(df_cell['region'].value_counts().ADS) + ' neurons)', 
        rotation='vertical', color = SPEC.COLS['ADS'],
        fontsize = 8)
fig.text(0.5, -0.02, 'Time from stimulus end [ms]', ha='center')
axs[0].set_title('Correct left trial')
axs[1].set_title('Correct right trial')
plt.tight_layout()
savethisfig(SPEC.FIGUREDIR + "figure1/", "fig1_trial_rasters")



