import argparse

import utils
from utils import load_config, get_non_nan_idx
import time
from Trainer import Trainer
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import datetime

# torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train_model(config: dict):

    if 'seed' in config.keys():
        utils.seed_everything(config['seed'])

    fname = config['experiment']['state_dict_fname']


    dataloader, val_dataloader, _ = config['dataset'].get_data_loaders(config,
                                                                       limit_num_data_points_to=config['limit_num_data_points_to'],
                                                                       distinguish_flowing=config['distinguish_flowing'],
                                                                       use_initial_pos_as_feature=config['use_initial_pos_as_feature'],
                                                                       )

    one_example_batch, *other = next(iter(dataloader))  #(bs, c, t, h, w)



    model: nn.Module = config['model'](**config['model_params'], data_dim=config['data_dim'],
                            dynamic_channels=config['dynamic_channels'], pred_stepsize=config['pred_stepsize'],)
    device = config['device']


    if not one_example_batch.dtype == torch.cfloat:
        model.to(device)
        # initialize lazy layers by calling a fw pass:
        idx_to_keep = get_non_nan_idx(one_example_batch[:, :, 0:config['pred_stepsize']+1])
        one_example_batch = one_example_batch[idx_to_keep]
        for i, o in enumerate(other):
            o = o[idx_to_keep].to(device)
            other[i] = o

        with torch.no_grad():
            model(one_example_batch[:, :, 0].to(device), one_example_batch[:, :, config['pred_stepsize']].to(device), *other)
    else:
        # model.to('cpu')
        print('first running a batch on cpu because moving model to device failed!')
        with torch.no_grad():
            model(one_example_batch[0:1, :, 0], one_example_batch[0:1, :, 1], *other)
        model.to(device)

    if 'starting_weight_state_dict' in config.keys():
        starting_state_dict = config['starting_weight_state_dict']
        if starting_state_dict is not None:
            print(f'initializing model from state dict {starting_state_dict}')
            model.load_state_dict(torch.load(starting_state_dict))


    print(f'the model has {utils.count_parameters(model)} parameters.')
    opt = config['optimizer'](model.parameters(), **config['opt_kwargs'])


    if 'clip_reconstr_loss_to' in config.keys():
        clip_reconstr_loss_to = config['clip_reconstr_loss_to']
    else:
        clip_reconstr_loss_to = torch.inf

    if 'clip_grad_norm' in config.keys():
        clip_grad_norm = config['clip_grad_norm']
    else:
        clip_grad_norm = None

    start_from_epoch = config['start_from_epoch'] if 'start_from_epoch' in config.keys() else 0

    trainer = Trainer(model, config['loss_func'], opt, config['pred_stepsize'], config['num_kl_annealing_cycles'],
                      config['kl_increase_proportion_per_cycle'], config, config['try_use_wandb'], config['beta'], clip_reconstr_loss_to,
                      config['detach_bptt_grad_chain'], config['increase_rollout_epochs'], clip_grad_norm=clip_grad_norm,
                      full_config=config)

    epochs = config['num_epochs']
    training_strategy = config['training_strategy']

    print(f'will train for {epochs} epochs with {training_strategy} training.')
    if not os.path.exists('models/state_dicts'):
        os.makedirs(os.path.join('models', 'state_dicts'))

    save_path = os.path.join('models', 'state_dicts', fname)
    print(f'will save model state dict as {fname}')

    config['experiment']['state_dict_path'] = save_path

    train_losses, train_acc, val_losses, val_acc, best_state_dict, last_state_dict = trainer.train(
        dataloader, val_dataloader, epochs, device, training_strategy, save_fname=fname, start_from_epoch=start_from_epoch)


    print(f'all done, saved at state dict at {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training config')
    parser.add_argument('config', type=str, help='config file path')
    args, remaining_args = parser.parse_known_args()
    config = load_config(args.config).config_dict

    def parse_to_int_or_float(str):
        try:
            return int(str)
        except ValueError:
            return float(str)

    for arg in remaining_args:
        arg: str
        arg = arg.strip('-')
        k, v = arg.split('=')
        try:
            v = parse_to_int_or_float(v)
        except:
            v = v
        if k in config.keys():
            config[k] = v
            print(f'set {k} to {v} in main config!', flush=True)
        elif k in config['model_params'].keys():
            config['model_params'][k] = v
            print(f'set {k} to {v} in model_params config!', flush=True)
        elif k in config['opt_kwargs'].keys():
            config['opt_kwargs'][k] = v
            print(f'set {k} to {v} in optimizer parameters config!', flush=True)
        else:
            config[k] = v
            print(f'did not find {k} in main or model param config keys -- set {k} to {v} in main config nevertheless', flush=True)

        if k != 'data_directory' and k != 'starting_weight_state_dict':
            old_state_dict_fname = config['experiment']['state_dict_fname']
            config['experiment']['state_dict_fname'] = old_state_dict_fname[:-3] + f'--{k[:5]}{v}' + old_state_dict_fname[-3:]

    start = time.time()
    print(f'starting new model training run')

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    config['experiment']['time'] = now_str
    train_model(config)
    stop = time.time()
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    print(f'finished at {now_str}. Total time: {np.round((stop - start) / 60., 2)} minutes.')
