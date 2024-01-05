from omegaconf import OmegaConf
import torch
import itertools
import wandb

# Relative Imports
from scale_learn import evaluate_model, run_experiment


def run_sweep(
    shared_config,
    variable_config,
    sweep_name,
    project="scale_learning",
    entity="mbasting",
):
    # NEED TO USE DOT NOTATION
    sweep_configuration = {
        "method": "bayes",
        "name": sweep_name,
        "metric": {"goal": "maximize", "name": "val/acc.max"},
        "parameters": {},
    }
    parameters = {
        "sweep": {"value": True},
        "exp_name": {"value": sweep_name},
    }
    # Convert configs to dictionary
    variable_config = OmegaConf.to_container(variable_config)
    shared_config = OmegaConf.to_container(shared_config)

    # Fill config file
    for key, value in shared_config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                parameters[f"{key}.{nested_key}"] = {"value": nested_value}
        else:
            parameters[key] = {"value": value}

    for key, value in variable_config.items():
        # Check if we are dealing with a range variable param
        if "min" in value and "max" in value:
            parameters[key] = {"max": value["max"], "min": value["min"]}
        else:
            parameters[key] = {"values": value}

    sweep_configuration["parameters"] = parameters
    print(sweep_configuration)

    sweep_id = wandb.sweep(sweep_configuration, project=project, entity=entity)
    return sweep_id


def run_exp(shared_config, variable_config):
    # Config contains config values that will remain the same throughout the whole exp
    # Varialbe config contains list of values that that hyperparameter will take on

    # Need to find each 'leaf' in the dictionary and try all options
    # To try all options need itertools which combines

    # Convert variable config to dictionary
    variable_config = OmegaConf.to_container(variable_config)

    options = []
    for key, value in variable_config.items():
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
            else:
                print("Something Wrong")
                return

        # Combine with shared config
        temp_conf = OmegaConf.create(temp_config)
        temp_conf = OmegaConf.merge(shared_config, temp_conf)
        run_experiment(temp_conf)
        # cleanup
        torch.cuda.empty_cache()


if __name__ == "__main__":
    CMD_CFG = OmegaConf.from_cli()
    SHARED_CONFIG = CMD_CFG.shared
    VARIABLE_CONFIG = CMD_CFG.var
    if "sweep" in CMD_CFG.keys() and CMD_CFG.sweep:
        run_sweep(
            SHARED_CONFIG, VARIABLE_CONFIG, CMD_CFG.sweep_name, project=CMD_CFG.project
        )
    elif "test_multiple" in CMD_CFG.keys():
        test_dataset_loader, test_seed = None, None
        for key in ["entity", "project", "tags", "in_depth", "cluster"]:
            assert key in SHARED_CONFIG.keys()

        run_ids = VARIABLE_CONFIG.run_ids
        for run_id in run_ids:
            SHARED_CONFIG.run_id = run_id
            test_dataset_loader, test_seed = evaluate_model(
                SHARED_CONFIG, test_dataloader=test_dataset_loader, test_seed=test_seed
            )

    else:
        run_exp(SHARED_CONFIG, VARIABLE_CONFIG)
