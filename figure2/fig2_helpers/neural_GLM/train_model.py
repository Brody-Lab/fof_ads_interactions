import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import KFold
from glm import glm, glm_rr, glmDataset, glmRRDataset
from common_imports import to_t, from_t



def train_model(model, 
                dataset,
                hyperprm_search = True,
                cv_predictions = True,
                regularizer = None,
                regularization_range=None,
                num_epochs=5000,
                num_folds=5,
                num_repeats=10,
                num_bootstraps =500,
                verbose = False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.PoissonNLLLoss(
        log_input=False,
        full=True,
        eps=1e-04
    )
    
    model_prms = dict()
    orig_state_dict = deepcopy(model.state_dict())
    
    def model_output(dataset, ds, model):
        if type(dataset) is glmRRDataset:
            outputs = model(ds['stimulus'], ds['input_spikes'], ds['spikes'])
        else:
            outputs = model(ds['stimulus'], ds['spikes'])
        return outputs

    
    # FIRST DO HYPERPARAMETER SEARCH FOR REG STRENGTH
    if hyperprm_search is True:
        if regularization_range is None:
            regularization_range = [0.0]  # Default value

        best_reg_param = None
        best_avg_val_loss = float('inf')

        loop_obj = tqdm(regularization_range)
        for reg_param in loop_obj:
            loop_obj.set_description(f"Reg strength: {reg_param}")
            avg_val_loss = 0.0
            kf = KFold(n_splits=num_folds, shuffle=True)

            for train_index, val_index in kf.split(dataset):
                train_dataset = dataset[train_index]
                val_dataset = dataset[val_index]
                model.load_state_dict(orig_state_dict)
                for epoch in range(num_epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model_output(dataset,train_dataset, model)
                    loss = loss_function(outputs, train_dataset['spikes'])
                    if verbose:
                        if epoch % 1000 == 0:
                            print("Epoch {}: Test loss {}, Regularizartion {}".format(
                                epoch, 
                                loss, 
                                reg_param * regularizer(model)))
                    if reg_param > 0.0:
                        loss += reg_param * regularizer(model)
                    loss.backward()
                    optimizer.step()
                    assert torch.isfinite(loss)


                model.eval()
                with torch.no_grad():
                    outputs = model_output(dataset,val_dataset,model)
                    val_loss = loss_function(outputs, val_dataset['spikes']).item()
                    if verbose:
                        print("      Validation {}".format(val_loss))

                avg_val_loss += val_loss / len(val_index)

            avg_val_loss /= num_folds

            if avg_val_loss < best_avg_val_loss:
                best_avg_val_loss = avg_val_loss
                best_reg_param = reg_param

        model_prms['best_reg_param'] = best_reg_param
        model_prms['best_avg_val_loss'] = best_avg_val_loss
        print("found best CV param: {}".format(best_reg_param))
    else:
        best_reg_param = 0.0

        
    # RETRAIN THE MODEL SAVING PARAMETERS FOR EACH FOLD
    if cv_predictions is True:
        print("Fitting to get predictions now")
        model.train()
        for repeat in p['num_repeats']:
            model_prms[repeat] = dict()
            kf = KFold(n_splits=num_folds, shuffle=True)
            fold = 0
            for train_index, val_index in kf.split(dataset):
                print("Fold number: {}".format(fold))
                train_dataset = dataset[train_index]
                val_dataset = dataset[val_index]
                model.load_state_dict(orig_state_dict)
                # for epoch in tqdm(range(num_epochs), desc="Training"):
                for epoch in range(num_epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model_output(dataset,train_dataset,model)
                    loss = loss_function(outputs, train_dataset['spikes'])
                    if best_reg_param > 0.0:
                        loss += best_reg_param * regularizer(model)
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    outputs = model_output(dataset, val_dataset, model)
                    val_loss = loss_function(outputs, val_dataset['spikes']).item()

                fold += 1
                model_prms[repeat][fold] = dict()
                model_prms[repeat][fold]['val_index'] = val_index
                model_prms[repeat][fold]['state_dict'] = model.make_state_dict()
                model_prms[repeat][fold]['validation_loss'] = val_loss
                model_prms[repeat][fold]['nspikes'] = from_t(torch.sum(val_dataset['spikes']))
                
            

        
    # RETRAIN THE MODEL USING THE ENTIRE DATASET
    print("Fitting with the whole dataset now")
    model.train()
    model.load_state_dict(orig_state_dict)
    full_dataset = dataset[range(len(dataset))]
    loss_list = []
    # for epoch in tqdm(range(num_epochs), desc="Final training"):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model_output(dataset,full_dataset, model)
        loss = loss_function(outputs, full_dataset['spikes'])
        loss_list.append(from_t(loss))
        if best_reg_param > 0.0:
            loss += best_reg_param * regularizer(model)
        loss.backward()
        optimizer.step()
    model_prms['state_dict'] = model.make_state_dict()
    model_prms['loss'] = np.array(loss_list)
        
        
    # BOOTSTRAP PARAMETERS
    if num_bootstraps > 0:
        bootstrap_params = dict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                bootstrap_params[name] = []

        for _ in tqdm(range(num_bootstraps), desc="Bootstrap prms"):
            # Sample with replacement from the dataset
            bootstrap_indices = np.random.choice(len(dataset), len(dataset), replace=True)
            bs_dataset = dataset[bootstrap_indices] 
            model.load_state_dict(orig_state_dict)
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model_output(dataset, bs_dataset, model)
                loss = loss_function(outputs, bs_dataset['spikes'])
                if best_reg_param > 0.0:
                    loss += best_reg_param * regularizer(model)
                loss.backward()
                optimizer.step()

            # Store parameters of the trained model
            for name, param in model.named_parameters():
                if param.grad is not None:
                    bootstrap_params[name].append(from_t(param))

        model_prms['bootstrap_params'] = bootstrap_params

    return model_prms



def l2_regularizer(model):

    reg = 0
    for var in model.covariates:
        reg += torch.norm(getattr(model, var + '_w').weight)
        
    return 0.5 * reg