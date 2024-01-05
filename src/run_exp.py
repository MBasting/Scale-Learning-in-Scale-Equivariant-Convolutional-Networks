import itertools
import warnings

import torch
import wandb
from omegaconf import OmegaConf

from train_lightning import check


def run_exp(shared_config, variable_config):
    # Config contains config values that will remain the same throughout the whole exp
    # Variable config contains list of values that that hyperparameter will take on

    # Need to find each 'leaf' in the dictionary and try all options
    
    # Convert variable config to dictionary
    variable_config = OmegaConf.to_container(variable_config)

    options = []
    for key, value in variable_config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                temp_options = []
                for el in nested_value:
                    temp_options.append([key, nested_key, el])
                options.append(temp_options)
        else:
            temp_options = []
            for el in value:
                temp_options.append([key, el])
            options.append(temp_options)

    for params in itertools.product(*options):
        temp_config = {}
        # Fill config file
        for param in params:
            if len(param) == 2:
                temp_config[param[0]] = param[1]
            elif len(param) == 3:
                # Check if parent folder already created
                if param[0] not in temp_config:
                    temp_config[param[0]] = {}
                temp_config[param[0]][param[1]] = param[2]
            else:
                print("Something Wrong")
                return

        # Combine with shared config
        temp_conf = OmegaConf.create(temp_config)
        temp_conf = OmegaConf.merge(shared_config, temp_conf)
        check(temp_conf)
        torch.cuda.empty_cache()

def run_sweep(shared_config, variable_config, project, sweep_name, entity='mbasting'):
    # NEED TO USE DOT NOTATION
    sweep_configuration = {
        'method': 'bayes',
        'name': sweep_name,
        'metric': {
            'goal': 'maximize',
            'name': 'Val accuracy Class.max'
        },
        'parameters': {}
    }
    parameters = {
        'wandb.sweep' : {'value' : True},
        'wandb.project': {'value' : project},
        'wandb.entity' : {'value' : entity},
        'wandb.local' : {'value' : False},
        'wandb.tags' : {'value' : ['Sweep', sweep_name]}
    }
    # Convert configs to dictionary
    variable_config = OmegaConf.to_container(variable_config)
    shared_config = OmegaConf.to_container(shared_config)

    # Fill config file
    for key, value in shared_config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                parameters[f'{key}.{nested_key}'] = {'value' : nested_value}
        else:
            parameters[key] = {'value' : value}

    for key, value in variable_config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                # Check if we are dealing with a range variable param
                if 'min' in nested_value and 'max' in nested_value:
                    parameters[f"{key}.{nested_key}"] = {'max': nested_value['max'], 'min' : nested_value['min']}
                else:
                    parameters[f"{key}.{nested_key}"] = {'values' : nested_value}
        else:
            # Check if we are dealing with a range variable param
            if 'min' in value and 'max' in value:
                parameters[key] = {'max': value['max'], 'min' : value['min']}
            else:
                parameters[key] = {'values' : value}
    
    sweep_configuration['parameters'] = parameters
    print(sweep_configuration)

    sweep_id = wandb.sweep(sweep_configuration,
                           project=project, entity=entity)
    return sweep_id



if __name__ == '__main__':
    CMD_CFG = OmegaConf.from_cli()
    SHARED_CONFIG = CMD_CFG.shared

    VARIABLE_CONFIG = CMD_CFG.var
    if 'sweep' in CMD_CFG.keys() and CMD_CFG.sweep:
        warnings.warn('Necessary Fields that you need are: model.index, dataset.index')
        run_sweep(SHARED_CONFIG, VARIABLE_CONFIG, CMD_CFG.project, CMD_CFG.sweep_name)
    elif 'test_multiple' in CMD_CFG.keys():
        SHARED_CONFIG.testing_mode = True
        test_dataset_loader, test_seed = None, None
        warnings.warn('Necessary Fields that you need are: wandb.local dataset.dynamic=True etc. ')
        run_ids = VARIABLE_CONFIG.run_ids
        for run_id in run_ids:
            SHARED_CONFIG.run_id = run_id
            test_dataset_loader, test_seed = check(SHARED_CONFIG, test_dataset_loader, test_seed) # Override test dataloader
        
    else:
        run_exp(SHARED_CONFIG, VARIABLE_CONFIG)
