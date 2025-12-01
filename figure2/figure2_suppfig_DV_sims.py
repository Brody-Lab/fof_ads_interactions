import sys, re
sys.path.insert(1, '../../figure_code/')
from my_imports import *
from helpers.phys_helpers import savethisfig

SIMDIR = SPEC.RESULTDIR + "figure2/DVsweeps/"
all_files = sorted(os.listdir(SIMDIR))

columns = ['num_neurons', 'num_trials', 'ff_delay', 'peak_time', 'accuracy_diff']
df = pd.DataFrame(columns = columns)

for f in all_files:
    p = np.load(SIMDIR + f, allow_pickle=True).item()
    data = {'num_neurons': p['params']['N_neurons_per_region'],
            'num_trials': p['params']['N_batch'],
            'ff_delay': p['params']['ff_delay'],
            'peak_time': p['DV_cc']['peak'],
            }
    
    df = df.append(data, ignore_index = True)
    
df['num_neurons'] = df['num_neurons'].astype(int)
df['num_trials'] = df['num_trials'].astype(int)
df['ff_delay'] = df['ff_delay'].astype(int)
df['peak_time'] = df['peak_time']*1000

grouped = df.groupby(['num_neurons', 'num_trials', 'ff_delay']).agg({'peak_time': ['mean', 'std']})
grouped = grouped.reset_index()

unique_ff_delays = grouped['ff_delay'].unique()
num_delays = len(unique_ff_delays)
fig, axs = plt.subplots(num_delays, figsize = (2, 1.5*num_delays), sharex = True, sharey = True)

for i, ff in enumerate(unique_ff_delays):
    filtered_df = grouped[grouped['ff_delay'] == ff]
    imshow_mean = filtered_df.pivot(index='num_trials', columns='num_neurons', values=('peak_time','mean'))
    imshow_std = filtered_df.pivot(index='num_trials', columns='num_neurons', values=('peak_time','std'))

    im = axs[i].imshow(imshow_mean, cmap='RdBu', aspect='auto', origin='lower', vmin = -50, vmax = 50)
    axs[i].set_title('True lag = {}ms'.format(-ff), fontsize = 8)
    axs[i].set_xticks(range(len(imshow_mean.columns)), imshow_mean.columns, fontsize = 8)
    axs[i].set_yticks(range(len(imshow_mean.index)), imshow_mean.index, fontsize = 8)

# Set common x and y labels
fig.text(0.5, 0.06, 'Number of neurons \n(per region)', ha='center', fontsize = 8)
fig.text(-0.1, 0.5, 'Number of trials', va='center', rotation='vertical', fontsize = 8)

# add the colorbar at the top of the plot
cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.015])  # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar_ax.set_title('Inferred lag \n(ms; peak of DV crosscorrelation)', fontsize = 8)
cbar.ax.tick_params(labelsize=8)

savethisfig(SPEC.FIGUREDIR + "figure2/", 'figure2_DVsweeps_lags')
