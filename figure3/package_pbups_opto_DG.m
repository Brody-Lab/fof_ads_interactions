% package_pbups_opto_DG parses opto traials correctly based on the settings
function[ratdata, avgdata] = package_pbups_opto_DG(ratnames, sessids, varargin)

% parsing the inputs
p = inputParser;

def_hit_thresh = 0.65;
def_tnum_thresh = 200;
def_lcorr_thresh = 0.4;
def_rcorr_thresh = 0.4;
def_left_flip = 0;
def_remove_violations = true;

addRequired(p,'ratnames', @iscell);
addRequired(p,'sessids', @isnumeric);
addParameter(p, 'hit_thresh', def_hit_thresh, @isnumeric);
addParameter(p, 'tnum_thresh', def_tnum_thresh, @isnumeric);
addParameter(p, 'lcorr_thresh', def_lcorr_thresh, @isnumeric);
addParameter(p, 'rcorr_thresh', def_rcorr_thresh, @isnumeric);
addParameter(p, 'left_flip', def_left_flip, @isnumeric);
addParameter(p, 'remove_violations', def_remove_violations);

parse(p, ratnames, sessids, varargin{:});

ratnames = p.Results.ratnames;
sessids = p.Results.sessids;
hit_thresh = p.Results.hit_thresh;
tnum_thresh = p.Results.tnum_thresh;
lcorr_thresh = p.Results.lcorr_thresh;
rcorr_thresh = p.Results.rcorr_thresh;
left_flip = p.Results.left_flip;
remove_violations = p.Results.remove_violations;

% if only one ratname is specified assume that all the sessions are from the same rat
if length(ratnames) == 1
    ratnames = repmat(ratnames, length(sessids),1);
    warning('Assuming all the sessions are from rat %s', ratnames{1})
end

[sessids, idx] = unique(sessids);
ratnames = ratnames(idx);

% ensuring that there are no duplicate sessions
if size(unique(sessids)) ~= size(sessids)
    warning('Some sessions are duplicates!!!!')
end

ratdata = [];

for i = 1:length(sessids)
    sessid = sessids(i);
    ratname = ratnames{i};
    
    if sessid == 708217
        continue;
    end
    
    fprintf('Processing rat %s, sessid %d \n', ratname, sessid);
    
    [ntrials, days, rcorr, lcorr, tcorr, sessidv, prot, ratnamev] = bdata( ...
        'select n_done_trials, sessiondate, right_correct, left_correct, total_correct, sessid, protocol, ratname from sessions where sessid="{S}"',sessid);
    
    if sessid ~= sessidv | ~strcmp(ratname, ratnamev) | isempty(sessidv)
        warning('session IDs and ratnames dont match, data for session %d (ratname %s) was not retrieved.', sessid, ratname)
        continue;
    end
    
    % thresholding the sessions
    if ntrials < tnum_thresh | lcorr < lcorr_thresh | rcorr < rcorr_thresh | tcorr < hit_thresh
        warning('session %d did not meet the thresholds, sorry!', sessid);
        continue;
    end
    
    S = get_sessdata(sessids(i));
    pd = S.pd{1};
    peh = S.peh{1};
    
    % if for some reason poke ins/outs have nans
    peh = correct_nan_pokes(peh);
    
    nTrials = min([structfun(@numel, pd); numel(peh)]);

    % some fields in "pd" have more elements than others
    pd = structfun(@(x) x(1:nTrials), pd, 'uni', 0);
    
    if length(pd.hits) ~= length(peh)
        continue;
    end
    
    if sum(cellfun(@(x) x.ison, pd.stimdata) ~= pd.stims) ~= 0
        warning('session %d had misligned opto stim fields, skipping!', sessid);
        continue;
    end
    
    
    % if the session had probe trials including just those
    % also check if the animals had cpoked in or side poked in
    [is_cpoke, is_probe] = deal([]);
    if isfield(pd.bupsdata{1}, 'is_probe_trial')
        exist_probe = 1;
        for tr = 1:length(peh)
            is_probe(tr) = pd.bupsdata{tr}.is_probe_trial;
            is_cpoke(tr) = ~isempty(peh(tr).states.cpoke1);			% if the rat never cpoked in
            is_movtime(tr) = ~isempty(peh(tr).states.wait_for_spoke);
        end
    else
        exist_probe = 0;
        for tr = 1:length(peh)
            is_probe(tr) = pd.bupsdata{tr}.real_T == 1;
            is_cpoke(tr) = ~isempty(peh(tr).states.cpoke1);
            is_movtime(tr) = ~isempty(peh(tr).states.wait_for_spoke);
        end
    end
    
    if (sum(is_probe) ~= 0) & (exist_probe)
        goodp = find(is_probe == 1);
    else
        goodp = [1:length(peh)]';
    end
    
    % removing violations
    violations = get_violations({peh.states});
    assert(sum(violations == pd.violations) == nTrials);
    if remove_violations
        goodv = find(pd.violations==0);    % non-violation trials
        goodm = find(is_movtime);
    else
        goodv = [1:length(peh)]';
        goodm = [1:length(peh)]';
    end
    
    % restrict the choice reporting time
    if remove_violations
        movtime = zeros(length(peh),1);
        for tr = 1:length(peh)
            if is_movtime(tr)
                movtime(tr) = diff(peh(tr).states.wait_for_spoke);
            end
        end
        goodmv = find(movtime>0.01 & movtime < 2.);
    else
        goodmv = [1:length(peh)]';
    end
    

    % okay select good trials
    goodc = find(is_cpoke);
    good = intersect(intersect(intersect(goodv, goodp), goodc), goodm);
    good = intersect(good, goodmv);

    a = length(ratdata);
 
    fprintf('Number of good trials %d %d %d %d %d\n', length(good), length(goodp), length(goodc), length(goodv), length(goodmv));
    
    for g = 1:length(good)
        
        id = good(g);
        
        ratdata(a+g).sessid = sessid;
        ratdata(a+g).ratname = ratnamev;
        ratdata(a+g).prot = prot;
        ratdata(a+g).date = days;
        
        ratdata(a+g).trial_type = 'a';  % accumulation trial
        if abs(pd.bupsdata{id}.gamma) == 99
            if pd.sides{id} ~= 'f'
                ratdata(a+g).trial_type = 's';    % side LED trial
            elseif pd.sides == 'f'
                ratdata(a+g).trial_type = 'f';   % free choice trial
            end
        end
        ratdata(a+g).is_frozen = pd.bupsdata{id}.is_frozen;
        ratdata(a+g).seed = pd.bupsdata{id}.seed;
        
        ratdata(a+g).sides = pd.sides(id);
        ratdata(a+g).hit = pd.hits(id);
        
        
        % reward location (important for free choice trials)
        left_reward = peh(id).states.left_reward; % reward times
        right_reward = peh(id).states.right_reward;
        if numel(left_reward) > 0
            ratdata(a+g).reward_loc = 'l';
        elseif numel(right_reward) > 0
            ratdata(a+g).reward_loc = 'r';
        else
            ratdata(a+g).reward_loc = ' '; % no reward
        end
        
        % poked right
        if ratdata(a+g).trial_type == 'f' % free choice trials
            ratdata(a+g).pokedR = ratdata(a+g).reward_loc == 'r';
        else % accumulation and side LED trials
            ratdata(a+g).pokedR = (pd.hits(id) == 1 & pd.sides(id) == 'r') | (pd.hits(id) == 0 & pd.sides(id) == 'l');
        end
        
        
        % processing stimulus
        ratdata(a+g).fbupstereo = pd.bupsdata{id}.first_bup_stereo;
        if pd.bupsdata{id}.first_bup_stereo == 1
            ratdata(a+g).nleft = pd.n_left(id)-1;
            ratdata(a+g).nright = pd.n_right(id)-1;
        elseif pd.bupsdata{id}.first_bup_stereo == 0
            ratdata(a+g).nleft = pd.n_left(id);
            ratdata(a+g).nright = pd.n_right(id);
        end
        ratdata(a+g).Delta = ratdata(a+g).nright - ratdata(a+g).nleft;
        ratdata(a+g).Sigma = ratdata(a+g).nleft + ratdata(a+g).nright;
        ratdata(a+g).gamma = pd.bupsdata{id}.gamma;
        ratdata(a+g).leftbups = pd.bupsdata{id}.left;
        ratdata(a+g).rightbups = pd.bupsdata{id}.right;
        
        ratdata(a+g).T = pd.bupsdata{id}.real_T;
        ratdata(a+g).sample = pd.samples(id);
        ratdata(a+g).mgap = pd.memory_gaps(id);												% memory gap (post-stimulus)
        if remove_violations
            ratdata(a+g).movtime = diff(peh(id).states.wait_for_spoke);
        else
            ratdata(a+g).violations = pd.violations(tr);
        end
        
        if strcmp(prot, 'PBups')
            ratdata(a+g).nic = diff(peh(id).states.cpoke1(1,:));
            ratdata(a+g).delay = ratdata(a+g).nic - ratdata(a+g).sample - ratdata(a+g).mgap;   % delay before stim start
        elseif strcmp(prot, 'PBupsDelay')
            ratdata(a+g).delay = pd.sdelays(id);
            ratdata(a+g).nic = ratdata(a+g).delay + ratdata(a+g).mgap + ratdata(a+g).sample;
        end
        
        
        
        if left_flip == 1
            
            nleftdummy = ratdata(a+g).nleft;
            leftbupsdummy = ratdata(a+g).leftbups;
            ratdata(a+g).nleft = ratdata(a+g).nright;
            ratdata(a+g).nright = nleftdummy;
            ratdata(a+g).Delta = -1.*ratdata(a+g).Delta;
            ratdata(a+g).leftbups = ratdata(a+g).rightbups;
            ratdata(a+g).rightbups = leftbupsdummy;
            ratdata(a+g).gamma = -1.*ratdata(a+g).gamma;
            ratdata(a+g).pokedR = ~ratdata(a+g).pokedR;
            
            if ratdata(a+g).sides == 'r'
                ratdata(a+g).sides = 'l';
            else
                ratdata(a+g).sides = 'r';
            end
            
            if ratdata(a+g).reward_loc == 'r'
                ratdata(a+g).reward_loc = 'l';
            else
                ratdata(a+g).reward_loc = 'r';
            end
        end
        
        
        
        if exist_probe
            ratdata(a+g).is_probe = pd.bupsdata{id}.is_probe_trial;
        else
            ratdata(a+g).is_probe = pd.bupsdata{id}.real_T == 1;
        end
        
        
        % default values which would be changed if opto was on during the trial
        ratdata(a+g).optoval = 0;
        ratdata(a+g).optodur = 0;
        ratdata(a+g).optotype = "na";
        %     % dealing with the misalignmnet b/w stim on which happened for some sessions
        %     % adding it in the beginning so that no info is added about this trial
        %     % change idu to id if you want to make it work again!
        %     if pd.stimdata{id}.ison ~= pd.stims(id)
        %         idu = id+1;
        %         if idu >= length(peh)
        %             break;
        %         end
        %         ratdata(a+g).opto_on = pd.stimdata{idu}.ison ~= 0;
        %     else
        %         idu = id;
        %         ratdata(a+g).opto_on = pd.stimdata{idu}.ison ~= 0;
        %     end
        
        assert(pd.stimdata{id}.ison == pd.stims(id))
        ratdata(a+g).opto_on = pd.stimdata{id}.ison ~= 0;
        if ratdata(a+g).opto_on
            if strcmp(pd.stimdata{id}.trigger, 'cpoke1')
                if pd.stimdata{id}.dur == 2
                    ratdata(a+g).optoval = 1; 				% 2 sec full trial
                    ratdata(a+g).optodur = pd.stimdata{id}.dur;
                    ratdata(a+g).optotype = "whole_trial";
                elseif pd.stimdata{id}.pre == 0 & round(pd.stimdata{id}.dur,3) == round(ratdata(a+g).delay,3)
                    ratdata(a+g).optoval = 2;				% pre stimulus
                    ratdata(a+g).optodur = pd.stimdata{id}.dur;
                    ratdata(a+g).optotype = "pre_stim";
                elseif round(pd.stimdata{id}.pre,2) == round(ratdata(a+g).delay,2) & round(pd.stimdata{id}.dur,3) == round(ratdata(a+g).T/2,3)
                    ratdata(a+g).optoval = 3;				% 1st half
                    ratdata(a+g).optodur = pd.stimdata{id}.dur;
                    ratdata(a+g).optotype = "1st_half";
                elseif round(pd.stimdata{id}.pre,2) == round(ratdata(a+g).delay + ratdata(a+g).T/2 ,2) & round(pd.stimdata{id}.dur,2) == round(ratdata(a+g).T/2,2)
                    ratdata(a+g).optoval = 4;				% 2nd half
                    ratdata(a+g).optodur = pd.stimdata{id}.dur;
                    ratdata(a+g).optotype = "2nd_half";
                elseif round(pd.stimdata{id}.pre,2) == round(ratdata(a+g).delay + ratdata(a+g).T,2 ) & round(pd.stimdata{id}.dur,3) == round(ratdata(a+g).mgap,3)
                    % 							if pd.stimdata{id}.dur ~=0
                    ratdata(a+g).optoval = 5;  				% memory period
                    ratdata(a+g).optodur = pd.stimdata{id}.dur;
                    ratdata(a+g).optotype = "memory";
                else warning('Unknown stimulation type 1, sessid %d trial %d', sessid, id); end
            elseif strcmp(pd.stimdata{id}.trigger, 'wait_for_spoke')
                ratdata(a+g).optoval = 6;						% movement epoch
                ratdata(a+g).optodur = pd.stimdata{id}.dur;
                ratdata(a+g).optotype = "movement";
            else warning('Unknown stimulation type 2, sessid %d trial %d', sessid, id)
            end
        end
        
        
    end
    
end

if ~isempty(ratdata)
    flds = fields(ratdata);
    for fl = 1:numel(flds)
        if sum(ismember({'leftbups', 'rightbups','sides','reward_loc', 'trial_type'}, flds{fl}))>0
            continue;
        end
        avgdata.(flds{fl}) = [ratdata.(flds{fl})];
    end
end

end



function violated = get_violations(states)
    % essentially repeats the code in RewardsSection that calculates
    % violations but corrently takes account of an extra condition
    % introduced by the temporary addition of cbreak states in 2022
    
    n_cpoke = cellfun(@(x)numel(x.cpoke1),states);
    temp_violation = cellfun(@(x)numel(x.temp_violation),states);
    violation_state = cellfun(@(x)numel(x.violation_state),states);    
    violated = temp_violation | violation_state | (n_cpoke>2 & ~isfield(states{1},'cbreak'));
    violated=violated(:);

end



function PEH = correct_nan_pokes(PEH)
    % replace NaN elements of pokes with sensible values based on the start
    % and end of the trial
    trial_start_time = get_trial_start_time(PEH);
    trial_end_time = get_trial_end_time(PEH);
    for i=1:numel(PEH)
        for port = ["C" "L" "R"]
            if ~isempty(PEH(i).pokes.(port))
                if isnan(PEH(i).pokes.(port)(1))
                    PEH(i).pokes.(port)(1)=trial_start_time(i);
                end
                if isnan(PEH(i).pokes.(port)(end))
                    PEH(i).pokes.(port)(end)=trial_end_time(i);
                end
                if any(isnan(PEH(i).pokes.(port)(:)))
                    error('wtf. only start and end pokes should be NaN I thought.');
                end
                pokes = PEH(i).pokes.(port)';
                if any(diff(pokes(:))<-1e-3) % allow 1ms of slop
                    error('pokes are not in temporal order.');
                end
            end
        end
    end
end


function trial_start_time = get_trial_start_time(PEH)
    for i=1:numel(PEH)
        starting_state = PEH(i).states.starting_state;
        trial_start_time(i) = min(PEH(i).states.(starting_state)(:));
    end
end

function trial_end_time = get_trial_end_time(PEH)
    for i=1:numel(PEH)
        ending_state = PEH(i).states.ending_state;
        trial_end_time(i) = max(PEH(i).states.(ending_state)(:));
    end
end