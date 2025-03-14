import torch
import torch.nn as nn
import numpy as np
from modules.NeCS import NeCS
import os
from datasets import Eindhoven_Dataset


config_dict = dict(
    ###### general ######
    device='cuda',
    ###### data ######
    dataset=Eindhoven_Dataset,
    data_directory=os.path.join('data', 'Eindhoven_station_data'),
    limit_num_data_points_to=np.inf,
    batch_size=16,  #64
    dynamic_channels=(0,1,2,3), #(0,1,2,3)
    data_dim=10,  # 4 for x and v, 4 for also type info about the node -- pedestrian vs exit vs geometry (not used) vs flowing -- 2 for initial pos
    pred_stepsize=10,  # data granularity is 0.1 s, so this corresponds to pred_stepsize*0.1 seconds
    use_initial_pos_as_feature=True,  # whether to use initial position as a feature for each pedestrian
    distinguish_flowing=True,  # whether to distinguish between flowing and non-flowing pedestrians
    use_vavg=True,  # whether to use the averaged velcoity over pred_stepsize steps (True) or the instantaneous velocity (False)

    ###### model ######
    model=NeCS,
    model_params=dict(
        # NeCS parameters
        num_layers=4,  # num layers in gnn modules
        emb_dim=128,  # hidden dim throughout the model
        nonlinearity=nn.GELU(),  # activation
        num_latent_dim=64,  # latent dim
        use_fixed_std=False,  # whether to use a reconstruction loss equiv. to MSE
        dist_thresh=5.,  # dist thresh (in m) for the edge construction
        sample_from_pred=False, # whether to sample from the predicted distribution (False during training)
        type_channels=(4, 5, 6, 7),  # channels that indicate node type (pedestrian, exit, geometry, flowing)
        recurrent_unit=nn.LSTM,  # nn.LSTM,
        apply_edge_correction_step=True,  # whether to apply a learned correction step along edges with neighbors
        rel_pos_loss_coef=1000.,  # multiplied by beta coefficient in the loss weighting, so actual coef is this * beta

        # VAE/GNN parameters
        free_bits=0.05,  # free bits parameter lambda
        attention=False,  # whether to use attention in the gnn
        agg_fn='sum',  # gnn aggregation function
        types=(1,2),  # 1-pedestrian, 2-virtual exit node
        scale_decoder_sigma=1.0,  # scale the sigma of the decoder (if not fixed std)
    ),



    ###### training etc ######
    loss_func=lambda pred_sample, x_true: torch.linalg.norm(pred_sample[..., (0,1)] - x_true[..., (0,1)], dim=(1,2)).sum(),  # func used to log proxy metrics -- not for training
    optimizer=torch.optim.Adam,
    opt_kwargs={'lr': 1e-4, 'weight_decay': 1e-4, 'betas': (0.9, 0.9), 'eps':1e-5},
    num_epochs=341,
    training_strategy='bptt',  # only option for NeCS: bptt for multi-step training
    detach_bptt_grad_chain=False,  # whether to do pushforward-stile training (True) or normal bptt-style training (False)
    increase_rollout_epochs=[51, 101, 151, 201, 226, 251, 276, 301], # when to increase the rollout length during training
    beta=0.01,  # beta in beta-vae
    clip_grad_norm=1.0,  # clip gradient norm to this value,
    kl_increase_proportion_per_cycle=100/341, # proportion of cycle time spent increasing beta (vs keeping it constant at beta)
    num_kl_annealing_cycles=1,  # number of KL annealing cycles
    try_use_wandb=True,  # set to false to not use WandB even if it is available

    ###### save some experiment output here ######
    experiment={'state_dict_fname': 'NeCS_training_run_weights.pt'},
)
