clear;

opdata = readtable('figure3/optodata_FOFSTR.xlsx',...
    'ReadVariableNames', true);

% not plotting separately for each hemisphere
left_flip = 1;
exclude = ~(opdata.probe ==0.6) & ~strcmp(opdata.ratname, 'X025') & ~strcmp(opdata.ratname, 'X038') & ~strcmp(opdata.ratname, 'X011');

c_col = [0 0 0];
o_col =  [237 177 32]./255;

saving = 1;
set(0,'DefaultFigureWindowStyle','normal')


%% indices of inactivations sessions, INACTIVATION DATA

fprintf("\n\nCollecting inactivation data \n")
idxR = find(strcmp(opdata.inact_side, 'r') & opdata.cntrl == 0 & exclude);
idxL = find(strcmp(opdata.inact_side, 'l') & opdata.cntrl == 0 & exclude);

% next lets gather all the data
[rdatR, adatR] = package_pbups_opto_DG(opdata.ratname(idxR), opdata.sess_id(idxR), 'left_flip', left_flip);
[rdatL, adatL] = package_pbups_opto_DG(opdata.ratname(idxL), opdata.sess_id(idxL));
adatL.hemi = adatL.ratname + "left";
adatR.hemi = adatR.ratname + "right";
if left_flip
    f = fieldnames(adatL);
    for i = 1:length(f)
        adatL.(f{i}) = [adatL.(f{i}), adatR.(f{i})];
    end
    rdatL = [rdatL, rdatR];
end


%% CONTROL DATA

fprintf("\n\nCollecting control data \n")
idxR = find(strcmp(opdata.inact_side, 'r') & opdata.cntrl == 1 & exclude);
idxL = find(strcmp(opdata.inact_side, 'l') & opdata.cntrl == 1 & exclude);

% next lets gather all the data
[rdatR_ctrl, adatR_ctrl] = package_pbups_opto_DG(opdata.ratname(idxR), opdata.sess_id(idxR), 'left_flip', left_flip);
[rdatL_ctrl, adatL_ctrl] = package_pbups_opto_DG(opdata.ratname(idxL), opdata.sess_id(idxL));
adatL_ctrl.hemi = adatL_ctrl.ratname + "left";
adatR_ctrl.hemi = adatR_ctrl.ratname + "right";
if left_flip
    f = fieldnames(adatL_ctrl);
    for i = 1:length(f)
        adatL_ctrl.(f{i}) = [adatL_ctrl.(f{i}), adatR_ctrl.(f{i})];
    end
    rdatL_ctrl = [rdatL_ctrl, rdatR_ctrl];
end

%% first plot whole region control vs opto psychometrics

adat = adatL;
figure('Color', 'white');

subplot(1,3,1)
OPTOVAL = 1;  % whole trial
ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL)));
[fitdata_c, fitdata_o] = plot_opto_psych(adat, ids, OPTOVAL);

subplot(1,3,2)
OPTOVAL = 3;  % first half
ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL)));
[fitdata_c, fitdata_o] = plot_opto_psych(adat, ids, OPTOVAL);

subplot(1,3,3)
OPTOVAL = 4;  % second half
ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL)));
[fitdata_c, fitdata_o] = plot_opto_psych(adat, ids, OPTOVAL);


%% PRINT STATS FOR WRITING

% inactivation
OPTOVALS = [1, 3, 4];

for val = 1:length(OPTOVALS)
    
    fprintf("\nInactivation:")
    print_stats(OPTOVALS(val), adatL)
    
    if OPTOVALS(val) == 1
        fprintf("\nControl: ")
        print_stats(OPTOVALS(val), adatL_ctrl)
    end
    
end





%% Now compute ipsi bias first for all rats for all different kinds of bootstrap

adat = adatL;

ratnames = unique(adat.hemi);
% ratnames = unique(adat.ratname);

OPTOVALS = [1];
bias = nan(length(OPTOVALS), length(ratnames));

for o = 1:length(OPTOVALS)
    OPTOVAL = OPTOVALS(o);
    r = 1;
    for rat = ratnames
        
        rat{1}
        ids_rat = cellfun(@(x) strcmp(x,rat{1}), adat.hemi);
        ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL))) & ids_rat;
        if sum(ids) > 0
            bias(o,r) = compute_mean_bias(adat, ids, OPTOVAL);
            [data_c, data_o] = compute_choice_statistics(adat, ids, OPTOVAL);
            [EBl{o,r}, BDl{o,r}] = computeCIbias(data_c, data_o);
            %             r = r+1;
        end
        r = r+1;
    end
end

% figure(); hold on;
% scatter(1*ones(length(ratnames),1), bias(1,:))
% scatter(1.1*ones(length(ratnames),1), bias(2,:))
% scatter(1.2*ones(length(ratnames),1), bias(3,:))
%
% xlim([0.5, 1.5])
% hline(0)



%% compute statistics across rats

adat = adatL;
OPTOVALS = [1,3,4];


for o = 1:length(OPTOVALS)
    OPTOVAL = OPTOVALS(o)
    ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL)));
    [data_c, data_o] = compute_choice_statistics(adat, ids, OPTOVAL);
    [EBl{o}, BDl{o}] = computeCIbias(data_c, data_o);
    % abEBl = abs(EBl);
end




%% plotting movement effects
figure('Name', 'Movement effects', 'Color', 'white'); hold on;
 
adat = adatL;
% get sessions with whole region inactivations
optos = adat.optoval==1;
opto_sessids =  unique(adat.sessid(optos));
ids = ismember(adat.sessid, opto_sessids);
 
subplot(1,2,1); hold on;
h1 = cdfplot(adat.movtime(ids & adat.optoval == 0 & adat.pokedR == 0));
set(h1, 'LineWidth', 1.5, 'Color', c_col);
h1 = cdfplot(adat.movtime(ids & adat.optoval == 1 & adat.pokedR == 0));
set(h1, 'LineWidth', 1.5, 'Color', o_col);
xlim([0.25 1.25]); xlabel('Movement time [s]'); ylabel('Cumulative probability')
title('Ipsi choices')
grid off;
set(gca, 'FontSize', 15)
set(gca, 'linewidth', 1.5)
set(gca,'TickDir','out')
 
subplot(1,2,2); hold on;
h1 = cdfplot(adat.movtime(ids & adat.optoval == 0 & adat.pokedR == 1));
set(h1, 'LineWidth', 1.5, 'Color', c_col);
h1 = cdfplot(adat.movtime(ids & adat.optoval == 1 & adat.pokedR == 1));
set(h1, 'LineWidth', 1.5, 'Color', o_col);
xlim([0.25 1.25]); xlabel('Movement time [s]');
set(gca, 'ytick', []);
set(gca, 'yticklabels', []);
set(gca, 'ylabel', []);
grid off;
set(gca, 'FontSize', 15)
set(gca, 'linewidth', 1.5)
set(gca,'TickDir','out')
 
title('Contra choices')
 
% Statistical analysis with median and IQR
% Ipsi choices
ipsi_control = adat.movtime(ids & adat.optoval == 0 & adat.pokedR == 0);
ipsi_opto = adat.movtime(ids & adat.optoval == 1 & adat.pokedR == 0);

ipsi_control_median = median(ipsi_control);
ipsi_control_iqr = iqr(ipsi_control);
ipsi_opto_median = median(ipsi_opto);
ipsi_opto_iqr = iqr(ipsi_opto);

[p_ipsi, h_ipsi] = ranksum(ipsi_control, ipsi_opto, 'Tail', 'right');

fprintf('Ipsi choices:\n');
fprintf('Control: %.3f ± %.3f (median ± IQR)\n', ipsi_control_median, ipsi_control_iqr);
fprintf('Opto: %.3f ± %.3f (median ± IQR)\n', ipsi_opto_median, ipsi_opto_iqr);
fprintf('P = %.4f, Mann-Whitney U test\n\n', p_ipsi);

% Contra choices
contra_control = adat.movtime(ids & adat.optoval == 0 & adat.pokedR == 1);
contra_opto = adat.movtime(ids & adat.optoval == 1 & adat.pokedR == 1);

contra_control_median = median(contra_control);
contra_control_iqr = iqr(contra_control);
contra_opto_median = median(contra_opto);
contra_opto_iqr = iqr(contra_opto);

[p_contra, h_contra] = ranksum(contra_control, contra_opto, 'Tail', 'right');

fprintf('Contra choices:\n');
fprintf('Control: %.3f ± %.3f (median ± IQR)\n', contra_control_median, contra_control_iqr);
fprintf('Opto: %.3f ± %.3f (median ± IQR)\n', contra_opto_median, contra_opto_iqr);
fprintf('P = %.4f, Mann-Whitney U test\n', p_contra);
 
set(gcf, 'Units', 'inches', 'Position', [3,2,7,5])
savethisfig(gcf(), 'figure1_mvttimes')


%% BOOTSTRAP CI
OPTOVALS = [1,3,4];
adat = adatL;

% Bootstrap parameters
n_bootstrap = 100; % Number of bootstrap iterations
alpha = 0.05; % For 95% CI

% Store results
bootstrap_results = struct();
opto_labels = {'Whole trial', 'First half', 'Second half'};
param_names = {'Lapse L', 'Lapse R', 'Sensitivity', 'Bias'};

fprintf('Running bootstrap analysis...\n');

for val = 1:length(OPTOVALS)
    OPTOVAL = OPTOVALS(val);
    
    % Get data for this condition
    ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL)));
    
    % Separate control and opto trials
    control_trials = ids & adat.optoval == 0;
    opto_trials = ids & adat.optoval == OPTOVAL;
    
    if sum(control_trials) < 100 || sum(opto_trials) < 100
        fprintf('Warning: Too few trials for %s condition\n', opto_labels{val});
        continue;
    end
    
    fprintf('Processing %s condition (%d control, %d opto trials)...\n', ...
            opto_labels{val}, sum(control_trials), sum(opto_trials));
    
    % Initialize bootstrap storage
    bootstrap_control_params = zeros(n_bootstrap, 4);
    bootstrap_opto_params = zeros(n_bootstrap, 4);
    
    % Bootstrap loop
    for boot = 1:n_bootstrap
        if mod(boot, 100) == 0
            fprintf('  Bootstrap iteration %d/%d\n', boot, n_bootstrap);
        end
        
        % Bootstrap resample control trials
        control_idx = find(control_trials);
        boot_control_idx = control_idx(randi(length(control_idx), length(control_idx), 1));
        
        % Bootstrap resample opto trials  
        opto_idx = find(opto_trials);
        boot_opto_idx = opto_idx(randi(length(opto_idx), length(opto_idx), 1));
        
        % Create temporary data structure for control bootstrap
        boot_control_ids = false(size(adat.sessid));
        boot_control_ids(boot_control_idx) = true;
        
        % Create temporary data structure for opto bootstrap
        boot_opto_ids = false(size(adat.sessid));
        boot_opto_ids(boot_opto_idx) = true;
        
        try
            % Fit psychometric to bootstrapped control data
            [fitdata_c_boot, ~] = plot_opto_psych_bootstrap(adat, boot_control_ids, 0);
            
            % Fit psychometric to bootstrapped opto data  
            [~, fitdata_o_boot] = plot_opto_psych_bootstrap(adat, boot_opto_ids, OPTOVAL);
            
            if ~isempty(fitdata_c_boot) && ~isempty(fitdata_o_boot) && ...
               isfield(fitdata_c_boot, 'beta') && isfield(fitdata_o_boot, 'beta')
                
                % Store original parameters
                control_params = fitdata_c_boot.beta;
                opto_params = fitdata_o_boot.beta;
                
                % Modify right lapse: Right lapse = Left lapse + Right lapse
                % Assuming order is [LapseL, LapseR, Sensitivity, Bias]
                control_params(2) = control_params(1) + control_params(2);  % Right = Left + Right
                opto_params(2) = opto_params(1) + opto_params(2);           % Right = Left + Right
                
                bootstrap_control_params(boot, :) = control_params;
                bootstrap_opto_params(boot, :) = opto_params;
            else
                % If fit failed, use NaN
                bootstrap_control_params(boot, :) = NaN;
                bootstrap_opto_params(boot, :) = NaN;
            end
            
        catch
            % If fit failed, use NaN
            bootstrap_control_params(boot, :) = NaN;
            bootstrap_opto_params(boot, :) = NaN;
        end
    end
    
    % Remove failed fits
    valid_boots = ~any(isnan(bootstrap_control_params), 2) & ~any(isnan(bootstrap_opto_params), 2);
    bootstrap_control_params = bootstrap_control_params(valid_boots, :);
    bootstrap_opto_params = bootstrap_opto_params(valid_boots, :);
    
    fprintf('  Valid bootstrap fits: %d/%d\n', sum(valid_boots), n_bootstrap);
    
    if sum(valid_boots) < 100
        fprintf('Warning: Too few valid bootstrap fits for %s\n', opto_labels{val});
        continue;
    end
    
    % Calculate confidence intervals
    bootstrap_results(val).label = opto_labels{val};
    bootstrap_results(val).n_valid_boots = sum(valid_boots);
    
    for p = 1:4
        % Control parameter CI
        control_ci = prctile(bootstrap_control_params(:, p), [100*alpha/2, 100*(1-alpha/2)]);
        bootstrap_results(val).control_median(p) = median(bootstrap_control_params(:, p));
        bootstrap_results(val).control_ci(p, :) = control_ci;
        
        % Opto parameter CI
        opto_ci = prctile(bootstrap_opto_params(:, p), [100*alpha/2, 100*(1-alpha/2)]);
        bootstrap_results(val).opto_median(p) = median(bootstrap_opto_params(:, p));
        bootstrap_results(val).opto_ci(p, :) = opto_ci;
        
        % Difference CI
        param_diff = bootstrap_opto_params(:, p) - bootstrap_control_params(:, p);
        diff_ci = prctile(param_diff, [100*alpha/2, 100*(1-alpha/2)]);
        bootstrap_results(val).diff_median(p) = median(param_diff);
        bootstrap_results(val).diff_ci(p, :) = diff_ci;
        
        % Check if difference is significant (CI doesn't include 0)
        bootstrap_results(val).diff_significant(p) = diff_ci(1) > 0 || diff_ci(2) < 0;
    end
end

% Add explanation in the output
fprintf('\n=== BOOTSTRAP RESULTS (95%% CI) ===\n');
fprintf('Note: Right lapse parameter now represents total right lapse (original left + right)\n\n');

for val = 1:length(bootstrap_results)
    if ~isempty(bootstrap_results(val).label)
        fprintf('\n%s (%d valid bootstrap samples):\n', ...
                bootstrap_results(val).label, bootstrap_results(val).n_valid_boots);
        
        for p = 1:4
            control_med = bootstrap_results(val).control_median(p);
            control_ci = bootstrap_results(val).control_ci(p, :);
            opto_med = bootstrap_results(val).opto_median(p);
            opto_ci = bootstrap_results(val).opto_ci(p, :);
            diff_med = bootstrap_results(val).diff_median(p);
            diff_ci = bootstrap_results(val).diff_ci(p, :);
            sig_str = '';
            if bootstrap_results(val).diff_significant(p)
                sig_str = ' *';
            end
            
            fprintf('  %s: Control %.3f [%.3f, %.3f], Opto %.3f [%.3f, %.3f], Diff %.3f [%.3f, %.3f]%s\n', ...
                    param_names{p}, control_med, control_ci(1), control_ci(2), ...
                    opto_med, opto_ci(1), opto_ci(2), diff_med, diff_ci(1), diff_ci(2), sig_str);
        end
    end
end

%% Plot results with bootstrap CIs
figure('Color', 'white');
colors = [170/255 187/255 205/255; 250/255 207/255 119/255; 136/255 163/255 113/255];

for p = 1:4
    subplot(1, 4, p); hold on;
    
    for val = 1:length(bootstrap_results)
        if ~isempty(bootstrap_results(val).label)
            
            % Control with bootstrap CI
            control_val = bootstrap_results(val).control_median(p);
            control_ci = bootstrap_results(val).control_ci(p, :);
            
            errorbar(1, control_val, control_val - control_ci(1), control_ci(2) - control_val, ...
                     'o', 'Color', 'k', 'MarkerFaceColor', colors(val,:), ...
                     'MarkerSize', 7, 'LineWidth', 0.8, 'CapSize', 0);
            
            % Opto with bootstrap CI
            opto_val = bootstrap_results(val).opto_median(p);
            opto_ci = bootstrap_results(val).opto_ci(p, :);
            
                % Connect with line (solid if significant, dashed if not)
            line_style = '-';
           
            
            line([1 2], [control_val opto_val], 'Color', 'k', ...
                 'LineWidth', 0.8, 'LineStyle', line_style);
            
            errorbar(2, opto_val, opto_val - opto_ci(1), opto_ci(2) - opto_val, ...
                     'o', 'Color', 'k', 'MarkerFaceColor', colors(val,:), ...
                     'MarkerSize', 7, 'LineWidth', 0.8, 'CapSize', 0);
            
        
            
            % Add significance marker
            if bootstrap_results(val).diff_significant(p)
                mid_x = 1.5;
                mid_y = (control_val + opto_val) / 2;
                text(mid_x, mid_y, '*', 'FontSize', 16, 'Color', colors(val,:), ...
                     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            end
        end
    end
    
    xlim([0.5 2.5]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'Control', 'Inactivation'});
    ylabel(sprintf('%s ± CI', param_names{p}), 'fontsize', 25); 
%     title(sprintf('%s', param_names{p}));
    grid off;
    set(gca, 'FontSize', 11);
    set(gca, 'linewidth', 1.5)
    set(gca,'TickDir','out')
    
    % Add horizontal line at zero
    y_lims = ylim;
    if y_lims(1) < 0 && y_lims(2) > 0
        line([0.5 2.5], [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1);
    end
end



set(gcf, 'Units', 'inches', 'Position', [2, 2, 16, 4]);

% Print detailed results
fprintf('\n=== BOOTSTRAP RESULTS (95%% CI) ===\n');
for val = 1:length(bootstrap_results)
    if ~isempty(bootstrap_results(val).label)
        fprintf('\n%s (%d valid bootstrap samples):\n', ...
                bootstrap_results(val).label, bootstrap_results(val).n_valid_boots);
        
        for p = 1:4
            control_med = bootstrap_results(val).control_median(p);
            control_ci = bootstrap_results(val).control_ci(p, :);
            opto_med = bootstrap_results(val).opto_median(p);
            opto_ci = bootstrap_results(val).opto_ci(p, :);
            diff_med = bootstrap_results(val).diff_median(p);
            diff_ci = bootstrap_results(val).diff_ci(p, :);
            sig_str = '';
            if bootstrap_results(val).diff_significant(p)
                sig_str = ' *';
            end
            
            fprintf('  %s: Control %.3f [%.3f, %.3f], Opto %.3f [%.3f, %.3f], Diff %.3f [%.3f, %.3f]%s\n', ...
                    param_names{p}, control_med, control_ci(1), control_ci(2), ...
                    opto_med, opto_ci(1), opto_ci(2), diff_med, diff_ci(1), diff_ci(2), sig_str);
        end
    end
end

savethisfig(gcf(), 'figure3_psychparams')

%% parameter tradeoff 
OPTOVALS = [1,3,4];
adat = adatL;

% Bootstrap parameters
n_bootstrap = 100;
alpha = 0.05;

% Store results
bootstrap_results = struct();
opto_labels = {'Whole trial', 'First half', 'Second half'};
param_names = {'Lapse L', 'Lapse R', 'Sensitivity', 'Bias'};

fprintf('Running bootstrap analysis with parameter tradeoffs...\n');

for val = 1:length(OPTOVALS)
    OPTOVAL = OPTOVALS(val);
    
    % Get data for this condition
    ids = ismember(adat.sessid, unique(adat.sessid(adat.optoval==OPTOVAL)));
    
    % Separate control and opto trials
    control_trials = ids & adat.optoval == 0;
    opto_trials = ids & adat.optoval == OPTOVAL;
    
    if sum(control_trials) < 100 || sum(opto_trials) < 100
        fprintf('Warning: Too few trials for %s condition\n', opto_labels{val});
        continue;
    end
    
    fprintf('Processing %s condition (%d control, %d opto trials)...\n', ...
            opto_labels{val}, sum(control_trials), sum(opto_trials));
    
    % Initialize bootstrap storage
    bootstrap_control_params = zeros(n_bootstrap, 4);
    bootstrap_opto_params = zeros(n_bootstrap, 4);
    
    % Bootstrap loop
    for boot = 1:n_bootstrap
        if mod(boot, 100) == 0
            fprintf('  Bootstrap iteration %d/%d\n', boot, n_bootstrap);
        end
        
        % Bootstrap resample control trials
        control_idx = find(control_trials);
        boot_control_idx = control_idx(randi(length(control_idx), length(control_idx), 1));
        
        % Bootstrap resample opto trials  
        opto_idx = find(opto_trials);
        boot_opto_idx = opto_idx(randi(length(opto_idx), length(opto_idx), 1));
        
        % Create temporary data structure for control bootstrap
        boot_control_ids = false(size(adat.sessid));
        boot_control_ids(boot_control_idx) = true;
        
        % Create temporary data structure for opto bootstrap
        boot_opto_ids = false(size(adat.sessid));
        boot_opto_ids(boot_opto_idx) = true;
        
        try
            % Fit psychometric to bootstrapped control data
            [fitdata_c_boot, ~] = plot_opto_psych_bootstrap(adat, boot_control_ids, 0);
            
            % Fit psychometric to bootstrapped opto data  
            [~, fitdata_o_boot] = plot_opto_psych_bootstrap(adat, boot_opto_ids, OPTOVAL);
            
            if ~isempty(fitdata_c_boot) && ~isempty(fitdata_o_boot) && ...
               isfield(fitdata_c_boot, 'beta') && isfield(fitdata_o_boot, 'beta')
                
                % Store original parameters
                control_params = fitdata_c_boot.beta;
                opto_params = fitdata_o_boot.beta;
                
                % Modify right lapse: Right lapse = Left lapse + Right lapse
                control_params(2) = control_params(1) + control_params(2);
                opto_params(2) = opto_params(1) + opto_params(2);
                
                bootstrap_control_params(boot, :) = control_params;
                bootstrap_opto_params(boot, :) = opto_params;
            else
                % If fit failed, use NaN
                bootstrap_control_params(boot, :) = NaN;
                bootstrap_opto_params(boot, :) = NaN;
            end
            
        catch
            % If fit failed, use NaN
            bootstrap_control_params(boot, :) = NaN;
            bootstrap_opto_params(boot, :) = NaN;
        end
    end
    
    % Remove failed fits
    valid_boots = ~any(isnan(bootstrap_control_params), 2) & ~any(isnan(bootstrap_opto_params), 2);
    bootstrap_control_params = bootstrap_control_params(valid_boots, :);
    bootstrap_opto_params = bootstrap_opto_params(valid_boots, :);
    
    fprintf('  Valid bootstrap fits: %d/%d\n', sum(valid_boots), n_bootstrap);
    
    if sum(valid_boots) < 100
        fprintf('Warning: Too few valid bootstrap fits for %s\n', opto_labels{val});
        continue;
    end
    
    % Calculate parameter differences for tradeoff analysis
    param_diffs = bootstrap_opto_params - bootstrap_control_params;
    
    % Calculate confidence intervals
    bootstrap_results(val).label = opto_labels{val};
    bootstrap_results(val).n_valid_boots = sum(valid_boots);
    
    for p = 1:4
        % Control parameter CI
        control_ci = prctile(bootstrap_control_params(:, p), [100*alpha/2, 100*(1-alpha/2)]);
        bootstrap_results(val).control_median(p) = median(bootstrap_control_params(:, p));
        bootstrap_results(val).control_ci(p, :) = control_ci;
        
        % Opto parameter CI
        opto_ci = prctile(bootstrap_opto_params(:, p), [100*alpha/2, 100*(1-alpha/2)]);
        bootstrap_results(val).opto_median(p) = median(bootstrap_opto_params(:, p));
        bootstrap_results(val).opto_ci(p, :) = opto_ci;
        
        % Difference CI
        diff_ci = prctile(param_diffs(:, p), [100*alpha/2, 100*(1-alpha/2)]);
        bootstrap_results(val).diff_median(p) = median(param_diffs(:, p));
        bootstrap_results(val).diff_ci(p, :) = diff_ci;
        
        % Check if difference is significant (CI doesn't include 0)
        bootstrap_results(val).diff_significant(p) = diff_ci(1) > 0 || diff_ci(2) < 0;
    end
    
    % === COMPUTE PARAMETER TRADEOFFS ===
    
    % Calculate correlation matrix of parameter changes
    [tradeoff_corr, tradeoff_pvals] = corr(param_diffs);
    bootstrap_results(val).tradeoff_corr = tradeoff_corr;
    bootstrap_results(val).tradeoff_pvals = tradeoff_pvals;
    
    % Store parameter differences for plotting
    bootstrap_results(val).param_diffs = param_diffs;
    
    % Calculate specific tradeoffs of interest
    tradeoff_pairs = {
        [1, 3], 'Lapse L vs Sensitivity';
        [2, 3], 'Lapse R vs Sensitivity';
        [1, 4], 'Lapse L vs Bias';
        [2, 4], 'Lapse R vs Bias';
        [3, 4], 'Sensitivity vs Bias';
        [1, 2], 'Lapse L vs Lapse R'
    };
    
    bootstrap_results(val).tradeoff_pairs = tradeoff_pairs;
    
    for i = 1:size(tradeoff_pairs, 1)
        p1 = tradeoff_pairs{i, 1}(1);
        p2 = tradeoff_pairs{i, 1}(2);
        
        r = tradeoff_corr(p1, p2);
        p_val = tradeoff_pvals(p1, p2);
        
        bootstrap_results(val).tradeoff_stats{i} = struct(...
            'pair', tradeoff_pairs{i, 2}, ...
            'correlation', r, ...
            'p_value', p_val, ...
            'significant', p_val < 0.05);
    end
end

%%

% === VISUALIZATION OF PARAMETER TRADEOFFS ===

% Create tradeoff correlation matrix plot
figure('Color', 'white');
n_conditions = length(bootstrap_results);
colors = [0.8 0.2 0.2; 0.2 0.6 0.8; 0.2 0.8 0.2];

for val = 1:n_conditions
    if ~isempty(bootstrap_results(val).label)
        subplot(1, n_conditions, val);
        
        % Plot correlation matrix
        corr_matrix = bootstrap_results(val).tradeoff_corr;
        pval_matrix = bootstrap_results(val).tradeoff_pvals;
        
        imagesc(corr_matrix, [-1 1]);
        colorbar;
        colormap(redblue); % or use: colormap(redbluecmap)
        
        % Add correlation values and significance
        for i = 1:4
            for j = 1:4
                if i ~= j
                    text_color = 'white';
                    if abs(corr_matrix(i,j)) < 0.5
                        text_color = 'black';
                    end
                    
                    sig_marker = '';
                    if pval_matrix(i,j) < 0.001
                        sig_marker = '***';
                    elseif pval_matrix(i,j) < 0.01
                        sig_marker = '**';
                    elseif pval_matrix(i,j) < 0.05
                        sig_marker = '*';
                    end
                    
                    text(j, i, sprintf('%.2f%s', corr_matrix(i,j), sig_marker), ...
                         'HorizontalAlignment', 'center', 'Color', text_color, ...
                         'FontSize', 10, 'FontWeight', 'bold');
                end
            end
        end
        
        set(gca, 'XTick', 1:4, 'XTickLabel', param_names, 'XTickLabelRotation', 45);
        set(gca, 'YTick', 1:4, 'YTickLabel', param_names);
        title(sprintf('%s\nParameter Tradeoffs', bootstrap_results(val).label));
        axis square;
    end
end

sgtitle('Parameter Change Correlations (* p<0.05, ** p<0.01, *** p<0.001)');
set(gcf, 'Units', 'inches', 'Position', [2, 2, 15, 5]);

% Create scatter plots for significant tradeoffs
figure('Color', 'white');
subplot_count = 0;

for val = 1:n_conditions
    if ~isempty(bootstrap_results(val).label)
        param_diffs = bootstrap_results(val).param_diffs;
        
        % Find significant correlations
        sig_tradeoffs = [];
        for i = 1:length(bootstrap_results(val).tradeoff_stats)
            if bootstrap_results(val).tradeoff_stats{i}.significant
                sig_tradeoffs = [sig_tradeoffs, i];
            end
        end
        
        if ~isempty(sig_tradeoffs)
            for i = 1:min(3, length(sig_tradeoffs)) % Show up to 3 most significant
                subplot_count = subplot_count + 1;
                subplot(n_conditions, 3, subplot_count);
                
                tradeoff_idx = sig_tradeoffs(i);
                p1 = bootstrap_results(val).tradeoff_pairs{tradeoff_idx, 1}(1);
                p2 = bootstrap_results(val).tradeoff_pairs{tradeoff_idx, 1}(2);
                
                scatter(param_diffs(:, p1), param_diffs(:, p2), 30, colors(val,:), ...
                       'filled', 'MarkerFaceAlpha', 0.6);
                
                % Add regression line
                coeffs = polyfit(param_diffs(:, p1), param_diffs(:, p2), 1);
                x_line = linspace(min(param_diffs(:, p1)), max(param_diffs(:, p1)), 100);
                y_line = polyval(coeffs, x_line);
                hold on;
                plot(x_line, y_line, 'k-', 'LineWidth', 2);
                
                xlabel(sprintf('Δ%s', param_names{p1}));
                ylabel(sprintf('Δ%s', param_names{p2}));
                
                r = bootstrap_results(val).tradeoff_stats{tradeoff_idx}.correlation;
                p_val = bootstrap_results(val).tradeoff_stats{tradeoff_idx}.p_value;
                
                title(sprintf('%s\nr=%.3f, p=%.3f', bootstrap_results(val).label, r, p_val));
                grid on;
            end
        end
    end
end

sgtitle('Significant Parameter Tradeoffs');
set(gcf, 'Units', 'inches', 'Position', [2, 2, 15, 10]);

% Print tradeoff results
fprintf('\n=== PARAMETER TRADEOFF ANALYSIS ===\n');
for val = 1:length(bootstrap_results)
    if ~isempty(bootstrap_results(val).label)
        fprintf('\n%s:\n', bootstrap_results(val).label);
        fprintf('Parameter correlations (Pearson r, p-value):\n');
        
        for i = 1:length(bootstrap_results(val).tradeoff_stats)
            stats = bootstrap_results(val).tradeoff_stats{i};
            sig_str = '';
            if stats.significant
                sig_str = ' *';
            end
            
            fprintf('  %s: r = %.3f, p = %.4f%s\n', ...
                    stats.pair, stats.correlation, stats.p_value, sig_str);
        end
    end
end


%%
function [fitdata_c, fitdata_o] = plot_opto_psych_bootstrap(adat, ids, optoval)
    % Wrapper function for bootstrap - doesn't plot, just fits
    
    % Parameters matching your existing functions
    x_range = 40;
    x_binsize = 10;
    
    if optoval == 0
        % Fit control data only
        avgdata.pokedR = adat.pokedR(ids);
        avgdata.Delta = adat.Delta(ids);
        
        % Ensure we have enough data
        if length(avgdata.pokedR) < 50
            fitdata_c = [];
            fitdata_o = [];
            return;
        end
        
        fitdata_c = pbups_psych_gamma(avgdata,...
            'xreg', 'Delta',...
            'binwd_delta', x_binsize,...
            'range_delta', x_range,...
            'plotdata', false,...
            'ploterrorbar', false,...
            'plotfit', false);  % Don't plot anything for bootstrap
        
        fitdata_o = [];
        
    else
        % Fit opto data only
        avgdata.pokedR = adat.pokedR(ids);
        avgdata.Delta = adat.Delta(ids);
        
        % Ensure we have enough data
        if length(avgdata.pokedR) < 50
            fitdata_c = [];
            fitdata_o = [];
            return;
        end
        
        fitdata_o = pbups_psych_gamma(avgdata,...
            'xreg', 'Delta',...
            'binwd_delta', x_binsize,...
            'range_delta', x_range,...
            'plotdata', false,...
            'ploterrorbar', false,...
            'plotfit', false);  % Don't plot anything for bootstrap
        
        fitdata_c = [];
    end
end


%%


function[fitdata_c, fitdata_o] = plot_opto_psych_lite(adat, ids, OPTOVAL)

c_col = [220, 220, 220]./255;
o_col =  [250 207 143]./255;
x_range = 40;
x_binsize = 10;
x_ticknum = [-30:15:30];

% plot control trials
avgdata.pokedR = adat.pokedR(ids & adat.optoval == 0);
avgdata.Delta = adat.Delta(ids & adat.optoval == 0);
fitdata_c = pbups_psych_gamma(avgdata,...
    'axHandle', gca(),...
    'xreg', 'Delta',...
    'binwd_delta', x_binsize,...
    'range_delta', x_range,...
    'plotdata', false,...
    'ploterrorbar', false,...
    'dataFaceColor', c_col,...
    'fitLineColor', c_col,...
    'fitLineWidth', 1,...
    'dataLineColor', c_col,...
    'errorbarColor', c_col);

% plot opto trials
avgdata.pokedR = adat.pokedR(ids & adat.optoval == OPTOVAL);
avgdata.Delta = adat.Delta(ids & adat.optoval == OPTOVAL);
fitdata_o = pbups_psych_gamma(avgdata,...
    'axHandle', gca(),...
    'xreg', 'Delta',...
    'binwd_delta', x_binsize,...
    'range_delta', x_range,...
    'fitLineWidth', 1,...
    'plotdata', false,...
    'ploterrorbar', false,...
    'dataFaceColor', o_col,...
    'fitLineColor', o_col,...
    'dataLineColor', o_col,...
    'errorbarColor', o_col,...
    'xtickangle', 0,...
    'xlab', '#contra clicks - #ipsi clicks',...
    'ylab', 'Fraction chose contra',...
    'xticknum', x_ticknum);

end


%%
function[fitdata_c, fitdata_o] = plot_opto_psych(adat, ids, OPTOVAL)

c_col = 'k';
o_col =  [237 177 32]./255;
x_range = 40;
x_binsize = 10;
x_ticknum = [-30:15:30];

% plot control trials
avgdata.pokedR = adat.pokedR(ids & adat.optoval == 0);
avgdata.Delta = adat.Delta(ids & adat.optoval == 0);
fitdata_c = pbups_psych_gamma(avgdata,...
    'axHandle', gca(),...
    'xreg', 'Delta',...
    'binwd_delta', x_binsize,...
    'range_delta', x_range,...
    'dataFaceColor', c_col,...
    'fitLineColor', c_col,...
    'fitLineWidth', 1.8,...
    'dataLineColor', c_col,...
    'errorbarColor', c_col);

% plot opto trials
avgdata.pokedR = adat.pokedR(ids & adat.optoval == OPTOVAL);
avgdata.Delta = adat.Delta(ids & adat.optoval == OPTOVAL);
fitdata_o = pbups_psych_gamma(avgdata,...
    'axHandle', gca(),...
    'xreg', 'Delta',...
    'binwd_delta', x_binsize,...
    'range_delta', x_range,...
    'dataFaceColor', o_col,...
    'fitLineColor', o_col,...
    'fitLineWidth', 1.8,...
    'dataLineColor', o_col,...
    'errorbarColor', o_col,...
    'xtickangle', 0,...
    'xlab', '#contra clicks - #ipsi clicks',...
    'ylab', 'Fraction chose contra',...
    'xticknum', x_ticknum);

end

%%

function[data_c, data_o] = compute_choice_statistics(adat, ids, OPTOVAL)

range_delta = 40;
binwd_delta = 10;
edges = -range_delta - 0.25 : binwd_delta : range_delta + 0.25;
uxreg = round(edges(1:end-1) +  binwd_delta/2);

Delta = adat.Delta(ids & adat.optoval == OPTOVAL);
pokedR = adat.pokedR(ids & adat.optoval == OPTOVAL);
for i = 1:length(uxreg)
    ntrials(i) = sum(Delta < edges(i+1) & Delta > edges(i));
    fracR(i) = sum(pokedR(Delta < edges(i+1) & Delta > edges(i)))./ntrials(i);
end

Delta = adat.Delta(ids & adat.optoval == 0);
pokedR = adat.pokedR(ids & adat.optoval == 0);
for i = 1:length(uxreg)
    ntrials_ctrl(i) = sum(Delta < edges(i+1) & Delta > edges(i));
    fracR_ctrl(i) = sum(pokedR(Delta < edges(i+1) & Delta > edges(i)))./ntrials_ctrl(i);
end

data_c.ntrials = ntrials_ctrl;
data_c.fracR = fracR_ctrl;

data_o.ntrials = ntrials;
data_o.fracR = fracR;

end

function[ipsi_bias] = compute_mean_bias(adat, ids, OPTOVAL)

[data_c, data_o] = compute_choice_statistics(adat, ids, OPTOVAL);

GR = [data_o.fracR; data_c.fracR];
bad = sum(isnan(GR)) > 0;
GR(:,bad) = [];
GR = mean(GR,2);    % mean bias across delta clicks per each stim condition

ipsi_bias = [GR(2:end)-GR(1)]';

end

%%

function[] = print_stats(OPTOVAL, adat)

optos = adat.optoval==OPTOVAL;
opto_sessids =  unique(adat.sessid(optos));
ids = ismember(adat.sessid, opto_sessids);

fprintf("\n%s:\n ", adat.optotype(find(optos,1)));
fprintf("\t Number of trials %d\n", sum(ids));
fprintf("\t Number of sessions %d\n", length(opto_sessids));
fprintf("\t Number of rats %d\n", height(unique(cell2table(adat.ratname(optos)'))));
fprintf("\t Number of hemisphere %d\n", length(unique(adat.hemi(optos))));
end
