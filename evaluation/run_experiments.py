import sys
import os
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append('evaluation/')

from experiment_utils import *
from utils import set_size_plots
import torch
import pickle

from tueplots import bundles, figsizes
from utils import load_config
import experiment_functions as ef

plot_bundle = bundles.neurips2023()
width = 397.48499  # width in points
WIDTH = 397.48499  # width in points
plot_bundle['axes.labelsize'] = 10
plot_bundle['legend.fontsize'] = 10
plt.rcParams.update(plot_bundle)
figsize = figsizes.neurips2023(ncols=1)
num_fig_columns = 1.5  # 1.5
figsize['figure.figsize'] = set_size_plots(width, fraction=1 / num_fig_columns, h_to_w_ratio=None)
plt.rcParams.update(figsize)


@torch.no_grad()
def run_experiments(cfg):

    print('running experiments! \n\n')

    # ##### diagnostics based on crowd-level statistics:

    ## code for running simulations for initial conditions from the test set from scratch is commented out below:
    ## simulation data is also provided -- no need to run the below from scratch to make the validation plots!
    ## if you want to generate new simulations, uncomment the code below:

    #####

    # data_gt, data_sampled = ef.load_model_and_generate_sim(cfg, num_samples=np.inf,
    #                                                     use_posterior_sampling=False, use_train_set=False)
    # dict_to_save = {'data_gt': data_gt, 'data_sampled': data_sampled}
    # with open('traj_pedes_new_sim.pkl', 'wb') as handle:
    #     pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #####

    ## load simulation data for T^*, T^0 and additional isotropic repulsion variants, instead of running from scratch:
    with open('evaluation/traj_T_star.pkl', 'rb') as handle:  # T = T^* = 0.1
        dict_with_data = pickle.load(handle)
        data_gt = dict_with_data['data_gt']
        data_sampled = dict_with_data['data_sampled']
    with open('evaluation/traj_T_0.pkl', 'rb') as handle:  # T = T^0 = 0
        dict_with_data = pickle.load(handle)
        data_gt_frozen = dict_with_data['data_gt']
        data_sampled_frozen = dict_with_data['data_sampled']
    with open('evaluation/traj_isotr_repul.pkl', 'rb') as handle:  # additional isotropic repulsion
        dict_with_data = pickle.load(handle)
        data_sampled_colpush = dict_with_data['data_sampled']

    # # # tortuosity + autocorr
    ef.tortuosity(cfg, data_gt, [data_sampled, data_sampled_frozen],
                  autocorr_inset=2, exten='.svg')  # autocorr inset is the feat id (0-x, 1-y, 2-vx, 3-vy), or None

    # # # plot some trajectories
    ef.plot_trajectories(cfg,
                         [np.concatenate([d[..., :2] / 1000, d[..., 2:]], axis=-1) for d in data_gt],  # /1000 to go to m
                         [np.concatenate([d[..., :2] / 1000, d[..., 2:]], axis=-1) for d in data_sampled],
                         make_gifs=False,
                         num_plots=3,
                         ts_to_plot=(15, 30, 45), divide_extent_by=1000,  # to go to m
                         xlabel='', ylabel='', delete_first_axis=False, delete_second_axis=False,
                         use_simple_map=True, remove_ticks=True
                         )

    # # # nearest neighbor distribution and pdf over velocities + positions:
    ef.hist_and_polar(cfg, data_gt, data_sampled, max_radius=1.5, use_subplots=False,
                      filter_y_coor_smaller_than=5000.,
                      exten='.svg')
    ef.hist_and_polar(cfg, data_gt_frozen, data_sampled_frozen, max_radius=1.5, use_subplots=False,
                      filter_y_coor_smaller_than=5000.,
                      exten='_frozen.svg')

    # # # plot with mean and covariance of flow on the different locations on the map + histograms of pos in different segments
    ef.flow_fluct_and_pos_hist(cfg, data_gt, [data_sampled, data_sampled_frozen], prepend_to_title='test_')

    # # #  fundamental diagram:
    ef.fundamental_diagram(cfg, data_gt, data_sampled)

    # # # collision measurement (plot in paper appendix)
    ef.collisions(cfg, data_gt, [data_sampled, data_sampled_frozen, data_sampled_colpush], dists=[0.1, 0.25, 0.4, 0.55],
                  vel_diffs=[0.0, 0.25, 0.5, 0.75],
                  max_angle=180)

    # ################# virtual surrogate experiments:

    # in the paper, we used more repetitions for some of the experiments
    # we reduce the amount of repetitions here a bit when possible to reduce the computational load

    # pairwise avoidance:
    ef.counterflow_transfer_function(cfg, num_repetitions=3, delta_x_range=(-2500, 2500),  #TODO: 300
                                     tmax=11,
                                     corbetta_data="counterflow_transfer_function/Corbetta-2018-obs.h5")

    # # # force field for neighbors walking in the same/opposite directions:
    ## same:
    ef.force_scaling_neighbor_directions(cfg, delta_x_range=(-3500, 3500), vy=-134, num_repetitions=300,
                                         vel_mults=(1,), delta_y_range=(-3500, 3500), kind='same')
    ## opposite:
    ef.force_scaling_neighbor_directions(cfg, delta_x_range=(-3500, 3500), vy=-100, num_repetitions=300,
                                         vel_mults=(-1,), delta_y_range=(-3500, 0), kind='opposite')

    # # # # N-body scaling
    ef.force_scaling_crowd(cfg, num_repetitions=20,
                           num_y_vel=10, crowd_range=(1, 26), crowd_std=500,
                           get_crowd_as_lattice=True, crowd_com_dist_range=(-2500, -2000), crowd_lattice_width=2500,
                           y_vel_range=(-115., -70.), yrange=(50000, 75000), top_ks_social_forcing=(3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 20),
                           U0=0.375 * 0.8, R=2., exten='.svg'
                           )

    #
    # # # # scaling with gaussian distributed crowd:
    ef.force_scaling_crowd(cfg, num_repetitions=20, num_y_vel=10, crowd_range=(1, 26), crowd_std=500,
                           get_crowd_as_lattice=False, crowd_com_dist_range=(-3500, -2000), crowd_lattice_width=2000,
                           y_vel_range=(-115., -70.), yrange=(50000, 75000), top_ks_social_forcing=(3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 20),
                           U0=0.375 * 0.525 * 1.05, R=2., exten='.svg', append_to_title='_gaussian'
                           )

    # # N-body interaction with gap in the lattice:
    ef.force_scaling_crowd(cfg, num_repetitions=20, num_y_vel=10, crowd_range=(18, 19), crowd_std=500,
                           get_crowd_as_lattice=True, crowd_com_dist_range=(-2500, -2499), crowd_lattice_width=3000,
                           y_vel_range=(-115., -70.), yrange=(50000, 75000), top_ks_social_forcing=(3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 20),
                           U0=0.375 * 0.8, R=2.,
                           max_angle_sf=2*torch.pi,
                           add_spacing_center_lattice_range=(0., 4000.),
                           exten='.svg',
                           append_to_title='_spacing'
                           )
    ef.force_scaling_crowd(cfg, num_repetitions=20, num_y_vel=10, crowd_range=(18, 19), crowd_std=500,
                           get_crowd_as_lattice=True, crowd_com_dist_range=(-2500, -2499), crowd_lattice_width=3000,
                           y_vel_range=(-115., -70.), yrange=(50000, 75000), top_ks_social_forcing=(3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 20),
                           U0=0.375 * 0.8, R=2.,
                           max_angle_sf=torch.atan2(torch.tensor([2.0]),torch.tensor([4.])),
                           add_spacing_center_lattice_range=(0., 4000.),
                           exten='.svg',
                           append_to_title='_spacing_vision'
                           )

    return





if __name__ == '__main__':

    cfg = 'NeCS_experiment_default'
    state_dict = 'NeCS_state_dict.pt'
    print('----\n\nUSING CONFIG:', cfg, 'AND STATE DICT:', state_dict, '\n\n---')
    cfg = load_config(cfg, state_dict=state_dict).config_dict


    run_experiments(cfg)
