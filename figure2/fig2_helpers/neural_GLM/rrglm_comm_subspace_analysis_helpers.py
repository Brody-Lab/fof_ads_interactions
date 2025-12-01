from common_imports import *
from my_imports import *
from copy import deepcopy

from helpers.phys_helpers import get_sortCells_for_rat, datetime, split_trials
from helpers.physdata_preprocessing import load_phys_data_from_Cell
from helpers.rasters_and_psths import *
from collections import OrderedDict
from glmfits_utils import get_covar_dict


def load_glm_model(file, fit_name):

    model_dir = os.path.join(SPEC.RESULTDIR, 'neural_rr_GLM')
    with open(os.path.join(SPEC.RESULTDIR, 'optimal_rank_dict.pkl'), 'rb') as f:
        optimal_rank_dict = pickle.load(f)

    model_file = optimal_rank_dict[fit_name][file]['fit_filename']
    model_path = os.path.join(model_dir, model_file)

    print(f"Loading GLM file: {model_path}")
    summary = np.load(model_path, allow_pickle = True).item()
    state_dict = summary['model_dict'][fit_name]['state_dict']
    rank_dict = summary['rank_dict']
    p = summary['p']

    return model_path, state_dict, rank_dict, p


def load_trial_and_cell_data(file):

    files, datadir = get_sortCells_for_rat(file[:4], SPEC.DATADIR)
    df_trial, df_cell, _ = load_phys_data_from_Cell(os.path.join(datadir, file))
    df_cell = df_cell[df_cell.stim_fr > 1.0].reset_index(drop=True)

    return df_trial, df_cell



def reconstruct_full_kernels(state_dict, rank_dict):

    rank = rank_dict['rank']
    U = state_dict['rr_U_w.weight']
    V = state_dict['rr_V_w.weight']
    rr_basis_w = state_dict['rr_basis_w']
    basis_kernels = np.array(rank_dict['kernels']).T  # shape: (num_basis, kernel_len)

    num_basis, kernel_len = basis_kernels.shape
    basis_w = rr_basis_w.reshape(rank, num_basis)
    # Flip the weights to account for the flipped basis in the conv layer
    basis_w_flipped = np.flip(basis_w, axis = 1)

    # Reconstruct kernels for each rank component
    # Shape: (rank, num_basis, kernel_len)
    # Sum over basis functions: (rank, kernel_len)
    summed_kernels = (basis_w_flipped[:, :, np.newaxis] * basis_kernels[np.newaxis, :, :]).sum(axis = 1)

    num_outputs = V.shape[0]
    num_inputs = U.shape[1]
    full_kernels = np.zeros((num_outputs, num_inputs, kernel_len))

    for i in range(num_outputs):
        for j in range(num_inputs):
            for r in range(rank):
                # V[i,r] * U[r,j] gives the scalar coefficient for rank r
                # Multiply by the reconstructed kernel for rank r
                full_kernels[i,j] += V[i,r] * summed_kernels[r] * U[r,j]

    return full_kernels



def compute_susceptibility_influence(full_kernels):
    suscep = np.mean(np.abs(full_kernels), axis = (1, 2))   # output neurons
    influence = np.mean(np.abs(full_kernels), axis = (0, 2))        # input neurons
    return suscep, influence


def compute_percentile_labels(suscep, influence, df_cell, fit_name, percentile = 80):

    source_region, target_region = fit_name.split('_')
    df_out = df_cell[df_cell.region == target_region].reset_index(drop = True)
    df_in = df_cell[df_cell.region == source_region].reset_index(drop = True)

    sus_thresh_top = np.percentile(suscep, percentile)
    sus_thresh_bottom = np.percentile(suscep, 100 - percentile)
    inf_thresh_top = np.percentile(influence, percentile)
    inf_thresh_bottom = np.percentile(influence, 100 - percentile)

    records = []
    for i, val in enumerate(suscep):
        if val >= sus_thresh_top or val <= sus_thresh_bottom:
            records.append({
                'session': file,
                'direction': fit_name,
                'region': target_region,
                'cell_ID': df_out.loc[i, 'cell_ID'],
                'type': 'susceptibility',
                'value': val,
                'percentile': 'top' if val >= sus_thresh_top else 'bottom'
            })

    for i, val in enumerate(influence):
        if val >= inf_thresh_top or val <= inf_thresh_bottom:
            records.append({
                'session': file,
                'direction': fit_name,
                'region': source_region,
                'cell_ID': df_in.loc[i, 'cell_ID'],
                'type': 'influence',
                'value': val,
                'percentile': 'top' if val >= inf_thresh_top else 'bottom'
            })

    return pd.DataFrame(records)


def compute_kernel_metrics(state_dict, df_percentile, df_cell, p, fit_name):
    
    source_region, target_region = fit_name.split('_')
    df_target = df_cell.query('region == @target_region').reset_index(drop = True)
    df_topbottom = df_percentile.query('type == "susceptibility"').reset_index(drop = True)

    covar_dict = get_covar_dict(p)
    dt = p['binsize']/1000

    kernel_metrics = []

    for var in p['covariates']:

        covar_params = covar_dict[var]
        basis = covar_params['kernels'].T  # shape: (num_basis, time)
        kernel_length = basis.shape[1]
        t_axis = np.arange(kernel_length) * dt + covar_params['time_offset']

        weights = state_dict[f'{var}_w.weight']
        assert len(df_target) == weights.shape[0]

        for percentile in df_topbottom.percentile.unique():

            subset = df_topbottom.query('percentile == @percentile')

            for _, row in subset.iterrows():

                cell_ID = row['cell_ID']
                i = df_target[df_target['cell_ID'] == cell_ID].index[0]

                flipped_weights = np.flip(weights[i,:])
                kernel = flipped_weights @ basis  # shape: (time, )

                # compute metrics:

                # 1. total area under the kernel  (strength)
                total_area = np.sum(np.abs(kernel)) * dt
                
                # 2. peak amplitude
                peak_amplitude = np.max(np.abs(kernel))
                
                # 3. time to peak
                peak_idx = np.argmax(np.abs(kernel))
                time_to_peak = t_axis[peak_idx]

                # 4. time to 60% peak
                threshold_60 = 0.6 * peak_amplitude
                above_60_mask = np.abs(kernel) >= threshold_60
                if np.any(above_60_mask):
                    first_above_60 = np.argmax(above_60_mask)
                    last_above_60 = len(above_60_mask) - 1 - np.argmax(above_60_mask[::-1])
                    time_above_60 = t_axis[last_above_60] - t_axis[first_above_60]
                    time_to_60 = t_axis[first_above_60]
                else:
                    time_above_60 = 0
                    time_to_60 = np.nan

                kernel_metrics.append({
                    'session': row['session'],
                    'direction': fit_name,
                    'region': row['region'],
                    'cell_ID': cell_ID,
                    'type': 'susceptibility',
                    'percentile': percentile,
                    'covariate': var,
                    'total_area': total_area,
                    'peak_amplitude': peak_amplitude,
                    'time_to_peak': time_to_peak,
                    'time_to_60percent': time_to_60,
                    'time_above_60': time_above_60,
                    'kernel': deepcopy(kernel),
                    't_axis': deepcopy(t_axis)
                })

    return pd.DataFrame(kernel_metrics)



from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def bin_by_rate_helper(group, rate_col, n_rate_bins):

    group['rate_bin'] = pd.qcut(
        group[rate_col],
        q = n_rate_bins,
        labels = [f'bin_{i}' for i in range(n_rate_bins)],
        duplicates = 'drop'
    )

    return group


def compute_mannwhitney_effect(top, bottom, variable):

    u_stat, p_val = mannwhitneyu(top[variable], bottom[variable], alternative = 'two-sided')
    pooled = pd.concat([top[variable], bottom[variable]])
    effect_size = (np.median(top[variable]) - np.median(bottom[variable])) / np.std(pooled)
    return u_stat, p_val, effect_size


def _assemble_result_record(name, groupby_cols, rate_bin, data, top, bottom, stat, p, eff_size, variable, rate_col):

    if not isinstance(name, tuple):
        name = (name, )

    record = dict(zip(groupby_cols, name))
    if rate_bin is not None:
        record.update({
            'rate_bin': rate_bin,
            'rate_bin_min': data[rate_col].min(),
            'rate_bin_max': data[rate_col].max(),
            'rate_bin_median': data[rate_col].median(),
            'top_rate_median': np.median(top[rate_col]),
            'bottom_rate_median': np.median(bottom[rate_col])
        })

    record.update({
        'n_top': len(top),
        'n_bottom': len(bottom),
        f'{variable}_top_median': np.median(top[variable]),
        f'{variable}_bottom_median': np.median(bottom[variable]),
        'U_stat': stat,
        'p_value': p,
        'effect_size': eff_size
    })
    return record


def compare_groups_with_optional_binning(
    df,
    variable = 'total_area',
    groupby_cols = ['type', 'covariate'],
    percentile_col = 'percentile',
    top_label = 'top',
    bottom_label = 'bottom',
    rate_col = 'firing_rate',
    bin_by_rate = False,
    n_rate_bins = 4,
    min_n = 10
):

    results = []
    grouped = df.groupby(groupby_cols)

    for group_name, group in grouped:

        if bin_by_rate:

            group = bin_by_rate_helper(group, rate_col, n_rate_bins)
            if group is None:
                continue
            bin_levels = group['rate_bin'].dropna().unique()

            for rate_bin in bin_levels:
                bin_data = group[group['rate_bin'] == rate_bin]
                top = bin_data[bin_data[percentile_col] == top_label]
                bottom = bin_data[bin_data[percentile_col] == bottom_label]

                if len(top) < min_n or len(bottom) < min_n:
                    continue

                stat, p_val, eff_size = compute_mannwhitney_effect(top, bottom, variable)
                record = _assemble_result_record(group_name, groupby_cols, rate_bin, bin_data, top, bottom, stat, p_val, eff_size, variable, rate_col)
                results.append(record)
        else:
            top = group[group[percentile_col] == top_label]
            bottom = group[group[percentile_col] == bottom_label]

            if len(top) < min_n or len(bottom) < min_n:
                continue

            stat, p_val, eff_size = compute_mannwhitney_effect(top, bottom, variable)
            record = _assemble_result_record(group_name, groupby_cols, None, group, top, bottom, stat, p_val, eff_size, variable, rate_col)
            results.append(record)
    
    df_results = pd.DataFrame(results)

    if not df_results.empty:
        reject, p_adj, _, _ = multipletests(df_results['p_value'], alpha=0.05, method='bonferroni')
        df_results['p_adj'] = p_adj
        df_results['reject_fdr'] = reject

    return df_results

# df_simple = compare_groups_with_optional_binning(
#     df,
#     variable='auc',
#     groupby_cols=['kind', 'region'],
#     bin_by_rate=False
# )

# df_binned = compare_groups_with_optional_binning(
#     df_kernel_metrics,
#     variable='total_area',
#     groupby_cols=['type', 'covariate'],
#     bin_by_rate=True,
#     n_rate_bins=4,
#     min_n=10
# )


def plot_effects_grouped_by_covariate(
    df_results,
    variable='total_area',
    covariate_col='covariate',
    bin_col='rate_bin',
    effect_col='effect_size',
    pval_col='p_adj',
    significance_level=0.05,
    figsize=(12, 6),
    bar_width=0.15
):
    import matplotlib.pyplot as plt
    import numpy as np

    df_plot = df_results.copy()
    df_plot[bin_col] = df_plot[bin_col].astype(str)  # Ensure bins are categorical

    covariates = sorted(df_plot[covariate_col].unique())
    bins = sorted(df_plot[bin_col].unique())
    n_bins = len(bins)
    n_covs = len(covariates)

    x = np.arange(n_covs)
    offset_range = (np.arange(n_bins) - (n_bins - 1) / 2) * bar_width

    plt.figure(figsize=figsize)

    for i, bin_label in enumerate(bins):
        means = []
        sigs = []
        for cov in covariates:
            row = df_plot[(df_plot[covariate_col] == cov) & (df_plot[bin_col] == bin_label)]
            if not row.empty:
                means.append(row[effect_col].values[0])
                sigs.append(row[pval_col].values[0] < significance_level if pval_col in row else False)
            else:
                means.append(np.nan)
                sigs.append(False)

        bar_pos = x + offset_range[i]
        plt.bar(bar_pos, means, width=bar_width, label=f'{bin_label}')

        # Annotate stars
        for xpos, yval, is_sig in zip(bar_pos, means, sigs):
            if is_sig and not np.isnan(yval):
                plt.text(xpos, yval + 0.05 * np.sign(yval), '*', ha='center', va='bottom', fontsize=12)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(x, covariates, rotation=45)
    plt.xlabel('Covariate')
    plt.ylabel(f'Effect Size ({variable})')
    plt.title('Effect Sizes per Covariate Grouped by Rate Bin')
    plt.legend(title='Rate Bin')
    plt.tight_layout()
    plt.show()



from scipy.interpolate import interp1d

def plot_mean_kernels(
    df,
    kernel_col='kernel',
    t_col='t_axis',
    group_col='percentile',
    covariate_col='covariate',
    n_cols=3,
    figsize=(15, 6),
    show_sem=True,
    n_interp_points=150
):
    """
    Plot mean kernels (± SEM) from df_kernel_metrics, one subplot per covariate.
    Aligns using interpolation to a common time grid per covariate.
    """

    covariates = sorted(df[covariate_col].unique())
    n_rows = int(np.ceil(len(covariates) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, cov in enumerate(covariates):
        ax = axes[idx]
        df_sub = df[df[covariate_col] == cov]

        # Determine shared time grid (min-overlap approach)
        all_t = np.concatenate(df_sub[t_col].values)
        t_min, t_max = np.max([min(t) for t in df_sub[t_col]]), np.min([max(t) for t in df_sub[t_col]])
        if t_min >= t_max:
            print(f"⚠️ Skipping covariate {cov}: no overlap in time axes")
            continue
        t_common = np.linspace(t_min, t_max, n_interp_points)

        for label, group in df_sub.groupby(group_col):
            interpolated = []
            for k, t in zip(group[kernel_col], group[t_col]):
                f = interp1d(t, k, kind='linear', bounds_error=False, fill_value=np.nan)
                interp_kernel = f(t_common)
                interpolated.append(interp_kernel)
            interpolated = np.stack(interpolated)

            mean = np.nanmean(interpolated, axis=0)
            sem = np.nanstd(interpolated, axis=0) / np.sqrt(np.sum(~np.isnan(interpolated), axis=0))

            ax.plot(t_common, mean, label=label)
            if show_sem:
                ax.fill_between(t_common, mean - sem, mean + sem, alpha=0.3)

        ax.set_title(f'Covariate: {cov}')
        ax.axhline(0, linestyle='--', color='black', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response')
        ax.legend()

    # Clean up unused axes
    for j in range(len(covariates), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()



def analyze_glm_rr_session(file, fit_name):

    model_path, state_dict, rank_dict, p = load_glm_model(file, fit_name)
    df_trial, df_cell = load_trial_and_cell_data(file)

    full_kernels = reconstruct_full_kernels(state_dict, rank_dict)
    susceptibility, influence = compute_susceptibility_influence(full_kernels)
    
    df_percentile = compute_percentile_labels(susceptibility, influence, df_cell, fit_name)
    df_kernel_metrics = compute_kernel_metrics(state_dict, df_percentile, df_cell, p, fit_name)

    return df_percentile, df_cell, df_trial, df_kernel_metrics



all_df_sus = []
all_df_topbottom = []
all_kernel_metrics = []

fit_name = 'ADS_FOF'

for rat in SPEC.RATS:
    files, datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)

    for file in files:
        # --- Run analysis ---
        df_topbottom, df_cell, df_trial, df_kernel_metrics = analyze_glm_rr_session(file, fit_name)

        # --- Optional: Merge cluster info if `df` exists ---
        if 'df' in locals():
            df_cell = pd.merge(df_cell, df[['cell_ID', 'cluster']], on='cell_ID', how='left')

        # --- Store raw df_topbottom ---
        df_topbottom['rat'] = rat
        df_topbottom['session'] = file
        all_df_topbottom.append(df_topbottom)

        # --- Store kernel metrics ---
        df_kernel_metrics['rat'] = rat
        df_kernel_metrics['session'] = file
        all_kernel_metrics.append(df_kernel_metrics)

        # --- Extract and annotate susceptible/influential cells ---
        for kind in df_topbottom['type'].unique():
            for percentile in df_topbottom['percentile'].unique():
                filtered = df_topbottom[a
                    (df_topbottom['type'] == kind) &
                    (df_topbottom['percentile'] == percentile)
                ]

                df_sus = df_cell[df_cell['cell_ID'].isin(filtered['cell_ID'])].copy()
                df_sus['rat'] = rat
                df_sus['session'] = file
                df_sus['kind'] = kind
                df_sus['percentile'] = percentile

                all_df_sus.append(df_sus)

# --- Combine all results ---
combined_df_sus = pd.concat(all_df_sus, ignore_index=True)
combined_df_topbottom = pd.concat(all_df_topbottom, ignore_index=True)
combined_kernel_metrics = pd.concat(all_kernel_metrics, ignore_index=True)


combined_df_sus['auc_thresh'] = np.abs(combined_df_sus['auc'] - 0.5)
df_simple = compare_groups_with_optional_binning(
    combined_df_sus,
    variable='auc_thresh',
    groupby_cols=['kind', 'region'],
    bin_by_rate=False
)


combined_kernel_metrics = pd.merge(combined_kernel_metrics, combined_df_sus[['cell_ID', 'stim_fr', 'firing_rate']], 
                            on='cell_ID', how='left')
VARIABLE = 'total_area'

df_binned = compare_groups_with_optional_binning(
    combined_kernel_metrics,
    variable= VARIABLE,
    groupby_cols=['type', 'covariate'],
    bin_by_rate=True,
    n_rate_bins=3,
    min_n=10
)

plot_effects_grouped_by_covariate(
    df_results=df_binned,
    variable=VARIABLE,
    covariate_col='covariate',
    bin_col='rate_bin',
    pval_col='p_adj'
)


plot_mean_kernels(
    df=combined_kernel_metrics,
    kernel_col='kernel',
    t_col='t_axis',
    covariate_col='covariate',
    group_col='percentile',   # 'top' vs 'bottom'
    n_cols=3,
    figsize=(15, 8),
    show_sem=True
)
