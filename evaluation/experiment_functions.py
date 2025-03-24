import sys
sys.path.append('evaluation/')

from typing import List
from experiment_utils import *
import matplotlib.animation as animation
from utils import set_size_plots
from datetime import datetime
from scipy.optimize import curve_fit

import torch
import numpy as np
import time
import itertools
import pandas as pd
from tqdm import tqdm
from tueplots import bundles, figsizes
import matplotlib.image as mpimg
import matplotlib

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


def generate_sim(model, loader, num_samples=100, use_posterior_sampling=False, pred_stepsize=10,
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
    model.eval()
    model.to(device)

    all_data_gt = []
    all_data_sampled = []

    count = 0
    for data in loader:
        data, *miscellaneous_all = data
        batch_idx = miscellaneous_all[0]

        rollout_length = (data.shape[2] - 1) // pred_stepsize
        if verbose:
            print('starting model rollout')
        start_rollout_time = time.time()
        data = data.to(device)
        # rollout_length = 50
        trues, preds, _, dists = model_rollout(model, data, pred_stepsize,
                                               rollout_length, start_time=0,
                                               use_posterior_sampling=use_posterior_sampling,
                                               # True
                                               use_ground_truth_input=False, miscellaneous_all=[batch_idx],
                                               run_with_error_handling=False, return_dists=True)

        if verbose:
            print(
                f'{count} rollout of length {rollout_length} took {float(np.round(time.time() - start_rollout_time, 2))} seconds for {int(data.shape[0])} pedestrians')

        preds_samples = np.concatenate([i[:, :, None, ...] for i in preds], axis=2)  # (people, 1, time, feat)
        trues = np.concatenate([i[:, :, None, ...] for i in trues], axis=2)  # (people, 1, time, feat)

        preds_samples[..., :4] = utils.to_real_coordinates(preds_samples[..., :4], True,
                                                           False)
        trues[..., :4] = utils.to_real_coordinates(trues[..., :4], True, False)


        batch_idx = batch_idx.cpu().numpy()
        for b in np.unique(batch_idx):
            trues_b = trues[batch_idx == b]
            preds_samples_b = preds_samples[batch_idx == b]
            all_data_gt.append(trues_b)
            all_data_sampled.append(preds_samples_b)
            count += 1

        if count >= num_samples:
            break


    return all_data_gt, all_data_sampled


def load_model_and_generate_sim(cfg, num_samples=100, use_posterior_sampling=False, use_train_set=False):
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(
        cfg,
        True
    )

    loader = test_dataloader if not use_train_set else dataloader  # test_dataloader
    data_gt, data_sampled = generate_sim(model, loader, num_samples=num_samples,
                                         use_posterior_sampling=use_posterior_sampling, pred_stepsize=pred_stepsize)
    return data_gt, data_sampled




def plot_trajectories(cfg, data_gt, data_sampled=None, make_gifs=False, num_plots=10, append_to_title='',
                      prepend_to_title='', ts_to_plot=(-1,), xrange=None, yrange=None, delete_first_axis=False,
                      delete_second_axis=False,
                      divide_extent_by=1, xlabel=None, ylabel=None, use_simple_map=False, remove_ticks=False,
                      ):
    utils.seed_everything(123)


    figsize['figure.figsize'] = set_size_plots(width, fraction=2, h_to_w_ratio=None)

    assert len(data_gt) == len(data_sampled), 'data_gt and data_sampled must have the same length'

    plot_counter = 0
    idx_perm = np.random.permutation(len(data_gt))
    for i in range(len(data_gt)):
        num_people_total = data_gt[idx_perm[i]].shape[0]
        if num_people_total < 60:
            continue
        plot_counter += 1
        for t_up_to in ts_to_plot:
            trues = data_gt[idx_perm[i]][:, :, :t_up_to, :]
            preds_samples = data_sampled[idx_perm[i]][:, :, :t_up_to, :]

            # create two subplots: one for plotting the trajectories in trues, one for plotting the trajectories in the predictions of the model

            fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=False)

            bg_params = utils.plot_station_background(ax1, divide_extent_by=divide_extent_by)
            _ = utils.plot_station_background(ax2, divide_extent_by=divide_extent_by)

            y_col = 1
            x_col = 0

            _plot_pedestrians_on_axes(preds_samples, trues, ax1, ax2, y_col=y_col, x_col=x_col, markersize=4.,
                                      plot_last_t_seconds=10)

            if xrange is not None:
                ax1.set_ylim(*xrange)
                ax2.set_ylim(*xrange)
            if yrange is not None:
                ax1.set_xlim(*yrange)
                ax2.set_xlim(*yrange)

            if xlabel is not None:
                ax1.set_xlabel(xlabel)
                ax2.set_xlabel(xlabel)
            if ylabel is not None:
                ax1.set_ylabel(ylabel)
                ax2.set_ylabel(ylabel)

            if remove_ticks:
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])

            if delete_first_axis:
                fig.delaxes(ax1)
            if delete_second_axis:
                fig.delaxes(ax2)

            plot_fname = f"pedestrian_trajectories/{cfg['experiment']['state_dict_fname'][:-3]}"
            if use_simple_map:
                plot_fname = os.path.join(plot_fname, '_simple_map')
            plot_fname = os.path.join('figures/', prepend_to_title, plot_fname, append_to_title)
            if not os.path.exists(plot_fname):
                os.makedirs(plot_fname)

            plt.savefig(os.path.join(plot_fname, f"plot_{i}_t{t_up_to}.png"), bbox_inches='tight', dpi=600)

            plt.show()
        # plt.close()

        if make_gifs:
            trues = data_gt[idx_perm[i]]
            preds_samples = data_sampled[idx_perm[i]]
            # now make a gif where the pedestrians are moving and we track their trail as in the above figure:
            # let's use the funcanimation class from matplotlib to do this:

            fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)

            bg = mpimg.imread('map/station_map.png')

            def init():
                _ = utils.plot_station_background(ax1, divide_extent_by=divide_extent_by, bg=bg)
                _ = utils.plot_station_background(ax2, divide_extent_by=divide_extent_by, bg=bg)
                if xrange is not None:
                    ax1.set_xlim(*xrange)
                    ax2.set_xlim(*xrange)
                if yrange is not None:
                    ax1.set_ylim(*yrange)
                    ax2.set_ylim(*yrange)

                if xlabel is not None:
                    ax1.set_xlabel(xlabel)
                    ax2.set_xlabel(xlabel)
                if ylabel is not None:
                    ax1.set_ylabel(ylabel)
                    ax2.set_ylabel(ylabel)

                if delete_first_axis:
                    fig.delaxes(ax1)
                if delete_second_axis:
                    fig.delaxes(ax2)
                return ax1, ax2

            def animate(t):
                preds_samples_until_t = preds_samples[:, :, :t, :]
                trues_until_t = trues[:, :, :t, :]
                # clear the axes:
                ax1.cla()
                ax2.cla()
                utils.plot_station_background(ax1, divide_extent_by=divide_extent_by, bg=bg)
                utils.plot_station_background(ax2, divide_extent_by=divide_extent_by, bg=bg)
                _plot_pedestrians_on_axes(preds_samples_until_t, trues_until_t, ax1, ax2, y_col=y_col, x_col=x_col,
                                          markersize=4.,
                                      plot_last_t_seconds=10)
                if xrange is not None:
                    ax1.set_ylim(*xrange)
                    ax2.set_ylim(*xrange)
                if yrange is not None:
                    ax1.set_xlim(*yrange)
                    ax2.set_xlim(*yrange)

                if xlabel is not None:
                    ax1.set_xlabel(xlabel)
                    ax2.set_xlabel(xlabel)
                if ylabel is not None:
                    ax1.set_ylabel(ylabel)
                    ax2.set_ylabel(ylabel)

                if delete_first_axis:
                    fig.delaxes(ax1)
                if delete_second_axis:
                    fig.delaxes(ax2)
                return ax1, ax2

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=range(1, trues.shape[2]), interval=100, blit=False)

            path = f"pedestrian_gifs/{cfg['experiment']['state_dict_fname'][:-3]}"
            if use_simple_map:
                path = os.path.join(path, '_simple_map')
            path = os.path.join('figures',prepend_to_title, path)
            if not os.path.exists(path):
                os.makedirs(path)

            # Writer = animation.writers['ffmpeg']
            # writer = Writer(metadata=dict(artist='Kapitein Koenk'))
            fps = int(cfg['pred_stepsize'] * 0.1 * 10)  # 10 speedup factor

            fname = f"anim_sample_{i}_{append_to_title}.gif"
            if use_simple_map:
                fname = f"anim_sample_{i}_simple_map_{append_to_title}.gif"

            savepath = os.path.join(path, fname)
            anim.save(savepath, fps=fps,)

        if plot_counter >= num_plots:
            break

    print('all plots generated')



def _plot_pedestrians_on_axes(preds_samples, trues, ax1, ax2, y_col=1, x_col=0, markersize=1.,
                              plot_last_t_seconds=None):
    # get the non-pedestrian nodes (nodes where the feature with index 4 is not 1 at any point in time):
    non_pedestrian_nodes_pred = preds_samples[~np.any(preds_samples[..., 4] == 1, axis=(1, 2))]
    non_pedestrian_nodes_true = trues[~np.any(trues[..., 4] == 1, axis=(1, 2))]

    # filter the pedestrian nodes:
    preds_samples = preds_samples[np.any(preds_samples[..., 4] == 1, axis=(1, 2))]
    trues = trues[np.any(trues[..., 4] == 1, axis=(1, 2))]

    # plot the non-pedestrian nodes in black:
    # if non_pedestrian_nodes_true.shape[0] > 0:
    #     ax1.plot(non_pedestrian_nodes_pred[:, 0, 0, y_col], non_pedestrian_nodes_pred[:, 0, 0, x_col], 'o',
    #              color='black', markersize=markersize)
    #     ax2.plot(non_pedestrian_nodes_true[:, 0, 0, y_col], non_pedestrian_nodes_true[:, 0, 0, x_col], 'o',
    #              color='black', markersize=markersize)

    # plot the pedestrians:
    for i, p in enumerate(preds_samples):
        t = trues[i, 0]
        # if np.mean(np.isnan(t) > 0.3):
        #     continue
        # if np.mean(np.isnan(p) > 0.3):
        #     continue
        p = p[0]
        flowing = p[:, 7]
        if (flowing == 0).any():
            c = 'black'
            if plot_last_t_seconds is None:
                pl = ax1.plot(p[..., y_col], p[..., x_col], ':', color=c)
            else:
                pl = ax1.plot(p[-plot_last_t_seconds:, y_col], p[-plot_last_t_seconds:, x_col], ':', color=c)
            msize = markersize / 2
        else:
            if plot_last_t_seconds is None:
                pl = ax1.plot(p[..., y_col], p[..., x_col], ':')
            else:
                pl = ax1.plot(p[-plot_last_t_seconds:, y_col], p[-plot_last_t_seconds:, x_col], ':')
            msize = markersize
        c = pl[-1].get_color()
        pl = ax1.plot(p[-1, y_col], p[-1, x_col], 'o', color=c, markersize=msize)
        # c = pl[-1].get_color()
        # ax1.plot(p[-1, y_col], p[-1, x_col], 'o', color=c)
        if plot_last_t_seconds is None:
            pl = ax2.plot(t[..., y_col], t[..., x_col], ':',
                          color=c
                          )
        else:
            pl = ax2.plot(t[-plot_last_t_seconds:, y_col], t[-plot_last_t_seconds:, x_col], ':',
                          color=c
                          )
        c = pl[-1].get_color()
        ax2.plot(t[-1, y_col], t[-1, x_col], 'o',
                 color=c, markersize=msize
                 )


def flow_fluct_and_pos_hist(cfg, data_gt, data_sampled_list, prepend_to_title='', append_to_title='', density=True,
                            cb_loc='top', alpha=0.7):

    # construct a dataframe with the x, y coordinates as well as the vx, vy velocities:
    fullsim_gt = []
    for d in tqdm(data_gt):
        # get the velocities by differencing on positions:
        vel = np.diff(d[..., :2], axis=2, append=np.nan)
        d[..., 2:4] = vel
        # flatten time and people dims:
        d_flat = d.reshape(-1, d.shape[-1])
        # filter out NaNs:
        d_flat = d_flat[~np.isnan(d_flat).any(axis=1)]
        # keep only flowing:
        d_flat = d_flat[(d_flat[:, 7] == 1)]
        fullsim_gt.append(d_flat[:, :4] / 1000)  # go to m.

    fullsim_model = []
    for d in tqdm(data_sampled_list[0]):  # only the normal simulations, not the frozen
        # # get the velocities by differencing on positions:
        vel = np.diff(d[..., :2], axis=2, append=np.nan)
        d[..., 2:4] = vel
        # flatten time and people dims:
        d_flat = d.reshape(-1, d.shape[-1])
        # filter out NaNs:
        d_flat = d_flat[~np.isnan(d_flat).any(axis=1)]
        # keep only flowing:
        d_flat = d_flat[(d_flat[:, 7] == 1)]
        fullsim_model.append(d_flat[:, :4] / 1000)

    df_fullsim_gt = pd.DataFrame(np.vstack(fullsim_gt), columns=['x', 'y', 'vx', 'vy'])
    df_fullsim_model = pd.DataFrame(np.vstack(fullsim_model), columns=['x', 'y', 'vx', 'vy'])

    num_bins_x = 6
    num_bins_y = 20

    binned_x_gt, bins_x = pd.cut(df_fullsim_gt['x'], bins=num_bins_x, retbins=True)
    binned_y_gt, bins_y = pd.cut(df_fullsim_gt['y'], bins=num_bins_y, retbins=True)

    binned_x_model = pd.cut(df_fullsim_model['x'], bins=bins_x)
    binned_y_model = pd.cut(df_fullsim_model['y'], bins=bins_y)

    # calculate covariances:
    min_num_samples_per_loc = 200

    def mean_min_num_samples(arr):
        if len(arr) < min_num_samples_per_loc:
            return np.nan
        return np.mean(arr)

    grouped_fullsim_gt = df_fullsim_gt.groupby([binned_x_gt, binned_y_gt])
    agged_fullsim_gt = grouped_fullsim_gt.agg(mean_min_num_samples)
    grouped_fullsim_model = df_fullsim_model.groupby([binned_x_model, binned_y_model])
    agged_fullsim_model = grouped_fullsim_model.agg(mean_min_num_samples)

    covmat_fullsim_gt = df_fullsim_gt.groupby([binned_x_gt, binned_y_gt]).cov(min_periods=min_num_samples_per_loc)[
        ['vx', 'vy']]
    covmat_fullsim_gt = covmat_fullsim_gt[(covmat_fullsim_gt.index.get_level_values(None) == 'vx') | (
                covmat_fullsim_gt.index.get_level_values(None) == 'vy')]

    covmat_fullsim_model = \
    df_fullsim_model.groupby([binned_x_model, binned_y_model]).cov(min_periods=min_num_samples_per_loc)[['vx', 'vy']]
    covmat_fullsim_model = covmat_fullsim_model[(covmat_fullsim_model.index.get_level_values(None) == 'vx') | (
                covmat_fullsim_model.index.get_level_values(None) == 'vy')]

    ########
    # make the plot:
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    bg_params = utils.plot_station_background(ax, divide_extent_by=1000.)
    mean_norm = np.nanmean(np.linalg.norm(df_fullsim_gt[['vx', 'vy']].values, axis=-1), axis=0)

    scale_factor = 2.0  # 2.0

    for bin_x, bin_y in grouped_fullsim_gt.indices:

        x = bin_x.mid
        y = bin_y.mid
        try:
            # only plot if we also have sufficient model obs at the loc:
            vals_model_this = covmat_fullsim_model.loc[(bin_x, bin_y)].values
            if vals_model_this is None or np.isnan(vals_model_this).any():
                continue

            # plot the true cov ellipse:
            ellipse, _ = confidence_ellipse(0, 0, y, x, ax, n_std=scale_factor / mean_norm,
                                                             cov=covmat_fullsim_gt.loc[(bin_x, bin_y)].values,
                                                             facecolor='tab:red', alpha=alpha, swap_scale_x_and_y=True)
        except KeyError:
            continue

    for bin_x, bin_y in grouped_fullsim_model.indices:
        x = bin_x.mid
        y = bin_y.mid
        try:
            # only plot if we also have sufficient gt obs at the loc
            vals_gt_this = covmat_fullsim_gt.loc[(bin_x, bin_y)].values
            if vals_gt_this is None or np.isnan(vals_gt_this).any():
                continue
            # plot the model's cov ellipse:
            ellipse, _ = confidence_ellipse(0, 0, y, x, ax, n_std=scale_factor / mean_norm,
                                                             cov=covmat_fullsim_model.loc[(bin_x, bin_y)].values,
                                                             facecolor='tab:blue', alpha=alpha, swap_scale_x_and_y=True)
        except KeyError:
            continue

    # get the data for the matplotlib streamplot:
    def get_data_for_streamplot(bins_x, bins_y, agged_df):
        mid_x = bins_x[1:] - np.mean(np.diff(bins_x), keepdims=True) / 2
        mid_y = bins_y[1:] - np.mean(np.diff(bins_y), keepdims=True) / 2
        vy_grid = agged_df['vy'].values.reshape(mid_x.shape[0], mid_y.shape[0])
        vx_grid = agged_df['vx'].values.reshape(mid_x.shape[0], mid_y.shape[0])
        nan_mask = np.bitwise_or(np.isnan(vy_grid), np.isnan(vx_grid))
        return mid_x, mid_y, vx_grid, vy_grid, nan_mask

    mid_x_gt, mid_y_gt, vx_grid_gt, vy_grid_gt, nan_mask_gt = get_data_for_streamplot(
        bins_x, bins_y, agged_fullsim_gt)
    mid_x_model, mid_y_model, vx_grid_model, vy_grid_model, nan_mask_model = get_data_for_streamplot(
        bins_x, bins_y, agged_fullsim_model)
    nan_mask = np.bitwise_or(nan_mask_gt, nan_mask_model)
    # put any nan values to 0:
    vx_grid_gt[nan_mask] = 0
    vy_grid_gt[nan_mask] = 0
    vx_grid_model[nan_mask] = 0
    vy_grid_model[nan_mask] = 0

    # plotting parameters
    lwidth_streamplot = 5
    dens_streamplot = 0.3
    arrsize_streamplot = 4.

    # get the start points for streamplot integration:
    xmin = -0.7
    xmax = 0.7
    ymin = -0.7
    ymax = 0.7

    # make the stream plot:
    XX, YY = np.meshgrid(np.linspace(xmin, xmax, num_bins_x), np.linspace(ymin, ymax, num_bins_y))
    XX = XX.reshape(-1)
    YY = YY.reshape(-1)
    XY = torch.tensor(np.stack([XX, YY], axis=1)).float()[:, None, None, :]
    in_cafe_mask = utils._check_if_inside_cafe(
        utils.to_real_coordinates(
            torch.cat([XY, torch.zeros_like(XY)], dim=-1),  # function also expects velocity, simply put to 0
            False, True
        ),
        slack=7500,
        in_mm=True
    )[0][:, 0, 0]
    XY = XY[~in_cafe_mask]
    XY = utils.to_real_coordinates(
        torch.cat([XY, torch.zeros_like(XY)], dim=-1), False, True
        ) / 1000
    start_points_streamplot = XY[:,0,0,:2]
    start_points_streamplot = start_points_streamplot[:,(1,0)]  #swap x and y

    c = ax.streamplot(mid_y_gt, mid_x_gt, vy_grid_gt, vx_grid_gt, color='tab:red', broken_streamlines=True,
                      density=dens_streamplot, arrowsize=arrsize_streamplot, start_points=start_points_streamplot)
    c.lines.set_alpha(alpha)
    c.lines.set_linewidth(lwidth_streamplot)
    # Add markers to the lines
    add_markers_to_streamline(ax, c.lines, marker='$\odot$', marker_interval=8, markersize=12,
                              alpha=alpha)


    c = ax.streamplot(mid_y_model, mid_x_model, vy_grid_model, vx_grid_model, color='tab:blue', broken_streamlines=True,
                      density=dens_streamplot, arrowsize=arrsize_streamplot, start_points=start_points_streamplot)
    c.lines.set_alpha(alpha)
    c.lines.set_linewidth(lwidth_streamplot)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    path = 'figures/' + prepend_to_title + f"pedestrian_flow_fluct/{cfg['experiment']['state_dict_fname'][:-3]}"
    if not os.path.exists(path):
        os.makedirs(path)

    fname = os.path.join(path, f'flow_fluct_{append_to_title}.svg')
    # ax.set_xlabel('Ground truth')
    fig.savefig(fname, bbox_inches='tight', dpi=600, transparent=True)
    # plt.show()


    # add an inset hist of the x coorindate for two bins of the y-coordinates:
    # first construct a dataframe grouping on the y coor:
    # make a heatmap showing the positions of the pedestrians in both samples and ground-truth:
    all_gt = np.concatenate(data_gt, axis=0)
    all_sample_list = [np.concatenate(data_sampled, axis=0) for data_sampled in data_sampled_list]
    all_gt[..., :4] = all_gt[..., :4] / 1000.
    for all_sample in all_sample_list:
        all_sample[..., :4] /= 1000.

    all_gt[np.any(all_gt[..., 7] == 0, axis=(1, 2))] = np.nan
    # # only keep thos coors where the y-coor is greater than 5 to prevent the accumulation on the stairs dominating the vis:
    # all_gt[all_gt[..., 1] <= 5] = np.nan
    for all_sample in all_sample_list:
        all_sample[np.any(all_sample[..., 7] == 0, axis=(1, 2))] = np.nan
        all_sample[all_sample[..., 1] <= 5] = np.nan

    pos_gt = torch.from_numpy(all_gt[~np.isnan(all_gt).any(axis=-1)][..., :2])
    pos_sample_list = [torch.from_numpy(all_sample[~np.isnan(all_sample).any(axis=-1)][..., :2])
                       for all_sample in all_sample_list]


    # # inset plotting:

    # add an inset hist of the x coorindate for two bins of the y-coordinates:
    # first construct a dataframe grouping on the y coor:
    df_gt = pd.DataFrame({'x': torch.flatten(pos_gt[..., 0]).numpy(), 'y': torch.flatten(pos_gt[..., 1]).numpy()})
    bins = np.linspace(0, 80, 21)
    grouped_gt = df_gt.groupby(pd.cut(df_gt['y'], bins=bins))
    df_sample_list = [
        pd.DataFrame(
            {'x': torch.flatten(pos_sample_list[i][..., 0]).numpy(),
             'y': torch.flatten(pos_sample_list[i][..., 1]).numpy()}
        ) for i in range(len(pos_sample_list))
    ]
    grouped_sample_list = [df_sample.groupby(pd.cut(df_sample['y'], bins=bins)) for df_sample in df_sample_list]

    colors = ['tab:blue', 'tab:purple']
    linestyles = ['-', '--']
    labels = ['simulated', 'simulated ($T=0$)']
    new_fig, new_axs = plt.subplots(1, 3,
                                    figsize=set_size_plots(
        width=397, fraction= 1 / 2.25 * 9 / 10 * 9 / 5,
        h_to_w_ratio=(5 ** .5 - 1) / 2 * 10 / 9 * 5 / 9),
                                    sharex=True, sharey=True)


    for plot_counter, bin_id in enumerate([4, 9, 14]):
        grouped_gt_to_plot = grouped_gt
        for i, grouped_sample in enumerate(grouped_sample_list):
            new_ax = new_axs[plot_counter]
            make_hist_from_part_of_heatmap(ax, grouped_gt=grouped_gt_to_plot, grouped_sample=grouped_sample,
                                           bin_id=bin_id,
                                           new_ax=new_ax, color_sample=colors[i],
                                           linestyle_sample=linestyles[i],
                                           label_sample=labels[i], hist_bins=80)
            new_ax.set_ylim(bottom=3e-3, top=1e0)


        new_axs[1].set_xlabel('y [m]')
        new_axs[0].set_ylabel('PDF')

        new_fig.savefig(os.path.join(path, f'hists_flow_fluct_pos_all.svg'), bbox_inches='tight',
                        transparent=True
                        )



    fname = os.path.join(path, f'flow_fluct_w_hist_{append_to_title}.svg')
    fig.savefig(fname, bbox_inches='tight', dpi=600, transparent=True )
    plt.show()
    print('flow and pos hists generated')



def hist_and_polar(cfg, data_gt: List[torch.Tensor], data_sampled, max_radius=2., append_to_title='', prepend_to_title='',
                   use_subplots=True, calc_vel_from_pos_data=True, filter_y_coor_smaller_than=5000.,
                   require_sub_sampling_for_radar_plot=False, exten=None):

    width = 397
    num_fig_columns = 2 #1.5  # 1.5
    figsize['figure.figsize'] = set_size_plots(width, fraction=1 / num_fig_columns, h_to_w_ratio=None)
    plt.rcParams.update(figsize)

    if exten is None:
        exten='.svg'

    # initialize result objects
    all_pos_gt = []
    all_vel_gt = []
    all_pos_sample = []
    all_vel_sample = []
    all_nn_local_frame_coor_gt = []
    all_nn_local_frame_coor_sample = []


    for i in range(len(data_gt)):

        trues = np.ones_like(data_gt[i]) * data_gt[i]
        preds_samples = np.ones_like(data_sampled[i]) * data_sampled[i]
        pos_gt = trues[..., (0, 1)]
        pos_sample = preds_samples[..., (0, 1)]

        if calc_vel_from_pos_data: # use differencing for vol calc
            vel_gt = np.diff(pos_gt, axis=2) / 10  # /10 to go to cm/s
            vel_sample = np.diff(pos_sample, axis=2) / 10
            pos_gt = pos_gt[:, :, 1:]
            pos_sample = pos_sample[:, :, 1:]
        else:
            vel_gt = trues[..., (2, 3)]
            vel_sample = preds_samples[..., (2, 3)]

        flowing = (trues[..., 7] == 1).any(axis=(1, 2))  # which pedestrians are flowing towards the exit
        pedes = (trues[..., 4] == 1).any(axis=(1, 2))  # which nodes are pedestrians (as opposed to exit nodes)

        all_pos_gt.append(pos_gt[flowing])
        all_pos_sample.append(pos_sample[flowing])
        all_vel_gt.append(vel_gt[flowing])
        all_vel_sample.append(vel_sample[flowing])

        div_by = 1_000  # to got to m from mm

        # filter out pedestrians close to the exit for the polar plot:
        pos_gt[pos_gt[..., 1] < filter_y_coor_smaller_than] = np.nan
        pos_sample[pos_sample[..., 1] < filter_y_coor_smaller_than] = np.nan

        # store ground-truth data for polar:
        pos_gt_for_heatmap = np.transpose(pos_gt[flowing][:, 0], axes=(1, 0, 2)) / div_by
        vel_gt_for_heatmap = np.transpose(vel_gt[flowing][:, 0], axes=(1, 0, 2)) / div_by
        pos_gt_for_heatmap_incl_nonflowing = np.transpose(pos_gt[pedes][:, 0], axes=(1, 0, 2)) / div_by
        nn_local_frame_coor_gt = get_nn_coordinates_in_local_frame(pos_gt_for_heatmap,  # everything in m.
                                                                   vel_gt_for_heatmap,
                                                                   pos_gt_for_heatmap_incl_nonflowing,
                                                                   max_radius)
        all_nn_local_frame_coor_gt.append(nn_local_frame_coor_gt)

        # repeat for sampled data:
        pos_sample_for_heatmap = np.transpose(pos_sample[flowing][:, 0], axes=(1, 0, 2)) / div_by
        vel_sample_for_heatmap = np.transpose(vel_sample[flowing][:, 0], axes=(1, 0, 2)) / div_by
        pos_sample_for_heatmap_incl_nonflowing = np.transpose(pos_sample[pedes][:, 0], axes=(1, 0, 2)) / div_by

        nn_local_frame_coor_sample = get_nn_coordinates_in_local_frame(pos_sample_for_heatmap,
                                                                       vel_sample_for_heatmap,
                                                                       pos_sample_for_heatmap_incl_nonflowing,
                                                                       max_radius)
        all_nn_local_frame_coor_sample.append(nn_local_frame_coor_sample)

    pos_gt = np.concatenate(all_pos_gt, axis=0)
    pos_sample = np.concatenate(all_pos_sample, axis=0)
    vel_gt = np.concatenate(all_vel_gt, axis=0)
    vel_sample = np.concatenate(all_vel_sample, axis=0)

    # make a plot to compare the positions and velocity distributions of the model samples to the ground-truth
    # flatten to shape (obs, 2) for plotting:
    pos_gt = pos_gt.reshape(-1, 2)
    pos_gt = pos_gt[~np.isnan(pos_gt).any(axis=1)]
    pos_sample = pos_sample.reshape(-1, 2)
    pos_sample = pos_sample[~np.isnan(pos_sample).any(axis=1)]
    vel_gt = vel_gt.reshape(-1, 2)
    vel_sample = vel_sample.reshape(-1, 2)
    vel_gt = vel_gt[~np.isnan(vel_gt).any(axis=1)]
    vel_sample = vel_sample[~np.isnan(vel_sample).any(axis=1)]

    # plot the density plots:
    # create 2 rows and 2 columns of subplots:

    if use_subplots:  # everything in 4 subplots (used for logging)
        plot_densities_subplots(pos_gt, pos_sample, vel_gt, vel_sample, cfg, prepend_to_title,
                                append_to_title,
                                extension=exten)
    else:
        fname = os.path.join(f"pedes_hists", f"{cfg['experiment']['state_dict_fname'][:-3]}")
        fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
        if not os.path.exists(fname):
            os.makedirs(fname)

        #### y vel reversed pdf (plot for paper):
        fig, ax = plt.subplots(
            figsize=set_size_plots(width=397, fraction=1 / 2.25 * 9 / 10, h_to_w_ratio=(5 ** .5 - 1) / 2 * 10 / 9))
        r_yvel = (-0.5, 2.0)
        plot_density_single_plot_v2(-vel_gt[..., 1] / 100, -vel_sample[..., 1] / 100,
                                    bins_gt=20, bins_sampled=250, ax=ax, range=r_yvel)  # bins_gt=33
        plt.xlabel('v [m/s]')
        y, binedges = np.histogram(-vel_sample[..., 1] / 100, bins=250, range=r_yvel, density=True)
        idx_max = np.argmax(y)
        x_max = (binedges[idx_max] + binedges[idx_max + 1]) / 2
        ax.vlines(x_max, 0, y[idx_max], color='black', linestyle='--', zorder=-10, alpha=0.7)
        # Adding annotation
        mode_value = y[idx_max] - 0.05
        ax.annotate(f'{x_max:.2f}', (x_max, mode_value), textcoords="offset points", xytext=(0, 5), ha='center')
        # inset plot in log scale:
        left, bottom, width, height = [0.33, 0.5425, 0.225, 0.42]
        # left, bottom, width, height = [0.6, 0.45, 0.225, 0.5]
        ax_inset = fig.add_axes([left, bottom, width, height])
        plot_density_single_plot_v2(-vel_gt[..., 1] / 100, -vel_sample[..., 1] / 100,
                                    bins_gt=10, bins_sampled=100, ax=ax_inset, range=(-1.2, 2.5))  # bins_gt=17

        # inset gaussian fit:
        r = np.linspace(-1.2, 2.5, 1000)
        stats_for_gaussian = -vel_gt[..., 1] / 100
        stats_for_gaussian = stats_for_gaussian[(stats_for_gaussian > -1.2) & (stats_for_gaussian < 2.5)]
        ax_inset.plot(r, utils.gaussian_pdf(r, mu=np.mean(stats_for_gaussian), sigma=np.std(stats_for_gaussian)),
                      ':', color='black', alpha=0.75)
        ax_inset.set_yscale('log')
        ax_inset.set_ylim(1e-3, 5)
        ax.set_xlim(left=-0.75)
        ax.set_ylim(top=2.5)
        #
        ax.set_ylabel('PDF')
        ax.legend(ncols=2)
        ## save legend:
        legend = ax.get_legend()
        bbox = legend.get_window_extent()
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        fig_legend = plt.figure(figsize=(width, height))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        fig_legend.legend(handles=legend.legendHandles[::-1], labels=[t.get_text() for t in legend.get_texts()[::-1]],
                          loc='center',
                          ncols=2, frameon=False)
        fig_legend.savefig(os.path.join(fname, f"legend" + exten), bbox_inches='tight', transparent=True)
        # save hist:
        ax.get_legend().remove()
        fig.savefig(os.path.join(fname, f"yvel_reversed_paper" + exten), bbox_inches='tight', transparent=True)
        plt.show()
        print('done with hists')

    # now the polar plot:
    all_nn_local_frame_coor_gt = np.concatenate([i for i in all_nn_local_frame_coor_gt], axis=0)
    all_nn_local_frame_coor_sample = np.concatenate([i for i in all_nn_local_frame_coor_sample], axis=0)

    # we might limit the amount of points for the KDE density plot if it can go OOM:
    if require_sub_sampling_for_radar_plot and (all_nn_local_frame_coor_gt.shape[0] > 10000 or all_nn_local_frame_coor_sample.shape[0] > 10000):
        s = min(all_nn_local_frame_coor_gt.shape[0], all_nn_local_frame_coor_sample.shape[0])
        random_subset_idx = np.random.choice(
            s,
            size=min(s, 10000), replace=False
        )
        all_nn_local_frame_coor_gt = all_nn_local_frame_coor_gt[random_subset_idx]
        all_nn_local_frame_coor_sample = all_nn_local_frame_coor_sample[random_subset_idx]

    # make the plot:
    fig = plot_nn_heat_map(
        all_nn_local_frame_coor_gt,
        max_radius,
        show_plot=False,
        )

    fname = f"polar/{cfg['experiment']['state_dict_fname'][:-3]}"
    fname = os.path.join('figures',prepend_to_title, fname, append_to_title)
    if not os.path.exists(fname):
        os.makedirs(fname)

    plt.savefig(os.path.join(fname, f"polar_gt" + exten), bbox_inches='tight', transparent=True)


    fig = plot_nn_heat_map(
        all_nn_local_frame_coor_sample,
        max_radius,
        # title='Sample nearest neighbor KDE plot',
        show_plot=False,
        # xlabel='mu: ' + str(mu_gaussian_MLE_sample) + ' cov: ' + str(
        #     sigma_gaussian_MLE_sample) + ' KL: ' + str(KL)
        )
    cbar_ax = fig.axes[0]

    plt.savefig(os.path.join(fname, f"polar_sampled" + exten), bbox_inches='tight', transparent=True)

    # plot the color bar:
    fig, ax = plt.subplots(1,1)
    cbar = plt.colorbar(cbar_ax.get_images()[0], ax=ax)
    ax.remove()
    plt.savefig(os.path.join(fname, f"cbar" + exten), bbox_inches='tight', transparent=True)

    plt.show()

    print('all done!')


def fundamental_diagram(cfg, data_gt, data_sampled, prepend_to_title='', append_to_title='', save_as_png=False,
                        calc_vel_from_pos=True, exten='.svg'):

    figsize['figure.figsize'] = set_size_plots(width=397, fraction=1 / 2.25 * 9 / 10,
                                               h_to_w_ratio=(5 ** .5 - 1) / 2 * 10 / 9)
    plt.rcParams.update(figsize)

    list_with_densities_gt = []  # these lists will have length num_samples, and contain arrays of length rollout_length
    list_with_velocities_gt = []
    list_with_densities_sample = []
    list_with_velocities_sample = []

    for i in tqdm(range(len(data_gt))):
        trues = data_gt[i]
        preds_samples = data_sampled[i]

        pos_gt = trues[..., (0, 1)]  # (people, 1, time, feat)
        pos_sample = preds_samples[..., (0, 1)]
        if not calc_vel_from_pos:
            vel_gt = trues[..., (2, 3)]
            vel_sample = preds_samples[..., (2, 3)]
        else:
            vel_gt = np.diff(pos_gt, axis=2) / 10  # /10 to go to cm/s
            vel_sample = np.diff(pos_sample, axis=2) / 10
            pos_gt = pos_gt[:, :, 1:]
            pos_sample = pos_sample[:, :, 1:]
            trues = trues[:, :, 1:]
            preds_samples = preds_samples[:, :, 1:]

        # save data for plotting:
        # we calculate the local density with the calculate_local_density function from experiment utils using voronoi tesselation
        # for each person, we need:
        # - the local densities,
        # - the velocity

        # first calculate the local densities:
        # we first need to filter out the non-pedestrian nodes:
        pos_gt = pos_gt[np.any(trues[..., 4] == 1, axis=(1, 2))]
        pos_sample = pos_sample[np.any(preds_samples[..., 4] == 1, axis=(1, 2))]
        vel_gt = vel_gt[np.any(trues[..., 4] == 1, axis=(1, 2))]
        vel_sample = vel_sample[np.any(preds_samples[..., 4] == 1, axis=(1, 2))]

        # below expects data in mm:
        local_densities_gt = calculate_local_density(pos_gt, how='voronoi')  # (people, time)
        local_densities_sample = calculate_local_density(pos_sample, how='voronoi')

        # now filter out those pedestrians who did not move sufficiently to the negative y direction:
        trues_pedes = trues[np.any(trues[..., 4] == 1, axis=(1, 2))]
        flowing = trues_pedes[:, 0, :, 7] == 1

        local_densities_gt = local_densities_gt[flowing]
        local_densities_sample = local_densities_sample[flowing]
        vel_gt = vel_gt[:, 0][flowing]
        vel_sample = vel_sample[:, 0][flowing]

        # now get the speed:
        speed_gt = np.linalg.norm(vel_gt, axis=-1)
        speed_sample = np.linalg.norm(vel_sample, axis=-1)

        # finally, save in the lists:
        list_with_densities_gt.append(local_densities_gt)
        list_with_densities_sample.append(local_densities_sample)
        list_with_velocities_gt.append(speed_gt)
        list_with_velocities_sample.append(speed_sample)

    # put the results in an array:
    densities_gt_flat = np.concatenate(list_with_densities_gt, axis=0)
    velocities_gt_flat = np.concatenate(list_with_velocities_gt, axis=0)
    densities_sample_flat = np.concatenate(list_with_densities_sample, axis=0)
    velocities_sample_flat = np.concatenate(list_with_velocities_sample, axis=0)

    # filter out values where density is nan:
    velocities_gt_flat = velocities_gt_flat[~np.isnan(densities_gt_flat)] / 100  # to m/s
    densities_gt_flat = densities_gt_flat[~np.isnan(densities_gt_flat)]
    velocities_sample_flat = velocities_sample_flat[~np.isnan(densities_sample_flat)] / 100
    densities_sample_flat = densities_sample_flat[~np.isnan(densities_sample_flat)]

    # make the plot:
    import pandas as pd
    from matplotlib.patches import ConnectionPatch
    fig, ax1 = plt.subplots()  # figsize=(7, 3.5)
    xmax = 1.0
    step = 0.1
    # now for the model samples:
    df = pd.DataFrame(data=np.concatenate([densities_sample_flat[:, None], velocities_sample_flat[:, None]], axis=1),
                      columns=['density', 'velocity'])
    # groupby the num people and make a binned histogram
    groups = pd.cut(df.density, bins=np.arange(0, xmax, step))

    # now make a line plot with error bars for the mean and std of the velocity in each bin:
    def lower(x):
        return np.mean(x) - np.std(x)
    def upper(x):
        return np.mean(x) + np.std(x)

    df_grouped_samples = df.groupby(groups)
    df_agg_samples = df_grouped_samples.agg([np.mean, np.std, lambda x: lower(x), lambda x: upper(x)])

    # simulated data:
    ax1.plot(df_agg_samples['density']['mean'], df_agg_samples['velocity']['mean'], '-', color='tab:blue',
             label='simulated',
             zorder=1, linewidth=2)
    ax1.fill_between(df_agg_samples['density']['mean'],
                     df_agg_samples['velocity']['<lambda_0>'], df_agg_samples['velocity']['<lambda_1>'],
                     alpha=0.4, color='tab:blue', zorder=-1)

    # now for the ground-truth:
    df = pd.DataFrame(data=np.concatenate([densities_gt_flat[:, None], velocities_gt_flat[:, None]], axis=1),
                      columns=['density', 'velocity'])
    # groupby the num people and make a binned histogram
    bins = np.arange(0, xmax, step)
    groups = pd.cut(df.density, bins=bins)
    # now make a line plot with error bars for the mean and std of the velocity in each bin:
    df_grouped_gt = df.groupby(groups)
    df_agg_gt = df_grouped_gt.agg([np.mean, np.std, lambda x: lower(x), lambda x: upper(x)])
    error_bar_every = 2
    (_, caplines, _,) = ax1.errorbar(df_agg_gt['density']['mean'], df_agg_gt['velocity']['mean'],
                                     yerr=np.vstack(
                                         [df_agg_gt['velocity']['mean'] - df_agg_gt['velocity']['<lambda_0>'].values,
                                          df_agg_gt['velocity']['<lambda_1>'].values - df_agg_gt['velocity']['mean']]),
                                     errorevery=error_bar_every, marker=r'$\odot$', capsize=4, color='tab:red',
                                     markersize=8, label='observed', zorder=2, linewidth=0, elinewidth=2, capthick=2)
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_xlabel('Density [$m^{-2}$]')
    ax1.set_ylim(0.55, 1.7)
    fname = f"fundamental/sample_{cfg['experiment']['state_dict_fname'][:-3]}"
    fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
    if not os.path.exists(fname):
        os.makedirs(fname)
    plt.savefig(os.path.join(fname, f"fundamental" + exten), bbox_inches='tight', transparent=True)
    plt.show()
    print('fundamental diagram done')


def e1_correlation_function(cfg, data_gt, data_sampled, data_sampled_frozen=None, prepend_to_title='', append_to_title='', save_as_png=False,
                            calc_vel_from_pos=True, filter_y_coor_smaller_than=-np.inf, max_lag=20,
                            also_calc_grouped_on_dens=False):
    # copy the data for in-place editing
    data_gt = [np.ones_like(data_gt[i]) * data_gt[i] for i in range(len(data_gt))]
    data_sampled = [np.ones_like(data_sampled[i]) * data_sampled[i] for i in range(len(data_sampled))]
    if data_sampled_frozen is not None:
        data_sampled_frozen = [np.ones_like(data_sampled_frozen[i]) * data_sampled_frozen[i] for i in range(len(data_sampled_frozen))]

    if calc_vel_from_pos:
        for d_all in [data_gt, data_sampled, data_sampled_frozen]:
            if d_all is not None:
                for i, d in enumerate(d_all):
                    vel = np.diff(d[:, :, :, :2], axis=2)
                    d[:, :, 1:, 2:4] = vel
                    d = d[:, :, 1:]
                    d_all[i] = d

    # put all points with y coor smaller than the threshold to NaN:
    for d_all in [data_gt, data_sampled, data_sampled_frozen]:
        if d_all is not None:
            for i, d in enumerate(d_all):
                d[d[..., 1] < filter_y_coor_smaller_than] = np.nan
                d_all[i] = d

    if save_as_png:
        exten = '.png'
    else:
        exten = '.svg'


    feat_names = ['x', 'y', 'u', 'v']
    for feat_idx in range(4):
        corr_gt = calculate_auto_corr(data_gt, feat_idx=feat_idx, max_lag=max_lag)
        corr_sample = calculate_auto_corr(data_sampled, feat_idx=feat_idx, max_lag=max_lag)

        # now plot the results:
        fig = plt.figure()
        ax = plt.gca()
        # ax.plot(corr_gt, color='tab:blue', label='ground-truth')
        # ax.plot(corr_sample, color='tab:orange', label='model samples')
        plt.plot(range(len(corr_sample)), corr_sample, '-', color='tab:blue', label='simulated', markerfacecolor='none',
                 zorder=-1,
                 linewidth=2)
        plt.scatter(range(len(corr_gt)), corr_gt, marker=r'$\odot$', color='tab:red', label='observed',
                    facecolor='none', zorder=1, s=64)

        if data_sampled_frozen is not None:
            corr_sample_frozen = calculate_auto_corr(data_sampled_frozen, feat_idx=feat_idx, max_lag=max_lag)
            plt.plot(range(len(corr_sample_frozen)), corr_sample_frozen, '--', color='tab:purple', label='simulated (T0)', markerfacecolor='none',
                     zorder=-1,
                     linewidth=2)
        ax.set_xlabel('Lag [s]')
        ax.set_ylabel(f"Autocorrelation {feat_names[feat_idx]}")
        # ax.legend()
        fname = f"pedestrian_autocorr/{cfg['experiment']['state_dict_fname'][:-3]}"
        fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
        if not os.path.exists(fname):
            os.makedirs(fname)
        plt.savefig(os.path.join(fname, f"autocorr_{feat_names[feat_idx]}" + exten), bbox_inches='tight', transparent=True)
        plt.show()

    if also_calc_grouped_on_dens:
        # calculate the local densities for each point in data, at each time, and save as an additional feature at index 10
        all_dens = []
        for d_all in [data_gt, data_sampled]:
            for i, d in enumerate(d_all):
                dens = calculate_local_density(d[..., :2], how='voronoi')  # people, time
                all_dens.append(dens)
                d_and_dens = np.concatenate([d, dens[:, None, :, None]], axis=-1)
                d_all[i] = d_and_dens

        # bin the densities into density categories, and calculate the autocorrelation for each of the groups:
        bins = [0, 0.25, 0.5, 1000]
        labels = ['[0, 0.25)', '[0.25, 0.5)', '[0.5, inf)']
        for j, b in enumerate(bins[:-1]):
            low = b
            high = bins[j + 1]
            for feat_idx in range(2, 4):

                corr_gt = calculate_auto_corr(data_gt, feat_idx=feat_idx, max_lag=max_lag,
                                              min_avg_density=low, max_avg_density=high, all_densities=all_dens)
                corr_sample = calculate_auto_corr(data_sampled, feat_idx=feat_idx, max_lag=max_lag,
                                                  min_avg_density=low, max_avg_density=high, all_densities=all_dens)

                # now plot the results:
                fig = plt.figure()
                ax = plt.gca()
                # ax.plot(corr_gt, color='tab:blue', label='ground-truth')
                # ax.plot(corr_sample, color='tab:orange', label='model samples')
                plt.plot(range(len(corr_sample)), corr_sample, '-', color='tab:blue', label='simulated',
                         markerfacecolor='none',
                         zorder=-1,
                         linewidth=2)
                plt.scatter(range(len(corr_gt)), corr_gt, marker=r'$\odot$', color='tab:red', label='observed',
                            facecolor='none', zorder=1, s=64)
                ax.set_xlabel('Lag [s]')
                ax.set_ylabel(f"Autocorrelation {feat_names[feat_idx]}")
                ax.legend()
                fname = f"figures/pedestrian_autocorr/{cfg['experiment']['state_dict_fname'][:-3]}"
                fname = os.path.join(prepend_to_title, fname, append_to_title)
                if not os.path.exists(fname):
                    os.makedirs(fname)
                plt.title(f'density in {labels[j]}')
                plt.savefig(os.path.join(fname, f"autocorr_dens{labels[j]}_{feat_names[feat_idx]}" + exten),
                            bbox_inches='tight')
                plt.show()
    print('autocorrelation done')


def tortuosity(cfg, data_gt, data_sampled_list, prepend_to_title='', append_to_title='',
               autocorr_inset=None, calc_vel_from_pos=True, exten='.svg'):

    width = 397.48499  # width in points
    plt.rcParams.update(figsize)


    if calc_vel_from_pos:
        for d_all in [data_gt, *data_sampled_list]:
            for i, d in enumerate(d_all):
                vel = np.diff(d[:, :, :, :2], axis=2)
                d[:, :, 1:, 2:4] = vel
                d = d[:, :, 1:]
                d_all[i] = d


    # define the 4 café corners
    cafe_corners = [[28000, -500],
                    [50000, -700],
                    [50000, -5200],
                    [28000, -5000]]
    cafe_corners = np.flip(cafe_corners, axis=-1)
    # dists between corners -- inf for opposites
    r = cafe_corners.reshape(1, -1, 2) - cafe_corners.reshape(-1, 1, 2)
    cafe_dists = np.linalg.norm(r, axis=-1)
    cafe_dists[[0, 1, 2, 3], [2, 3, 0, 1]] = np.inf

    tort_gt = []
    tort_samp = [[] for _ in range(len(data_sampled_list))]
    tort_gt_dumb = []
    tort_samp_dumb = [[]for _ in range(len(data_sampled_list))]

    # save all pedestrian trajectories
    all_trajs_gt = []
    all_trajs_samp = [[] for _ in range(len(data_sampled_list))]

    # save the average densities in case we do binned on density calc:
    all_dens_gt = []
    all_dens_samp = [[] for _ in range(len(data_sampled_list))]

    # for each scenario trajectory
    for i in range(len(data_gt)):
        n_timesteps = data_sampled_list[0][i].shape[2]

        # first do this for ground truth, then for sampled
        for traj, tort, tort2, all_tr, all_dens in [
            [data_gt[i], tort_gt, tort_gt_dumb, all_trajs_gt, all_dens_gt],
            *[[data_sampled_list[j][i], tort_samp[j], tort_samp_dumb[j], all_trajs_samp[j], all_dens_samp[j]]
                for j in range(len(data_sampled_list))]  # all sampled simulations

        ]:
            # get rid of axis at 1 (size 1):
            traj = traj[:, 0]

            pos_incl_nonflowing = traj[..., (0, 1)]

            # select flowing only
            flowing = np.any(traj[..., 7] == 1, axis=-1)
            traj = traj[flowing]

            all_tr.extend(traj)

            # select position
            pos = traj[..., (0, 1)]  # (people, time, feat)

            # calculate the time where people enter the system (only look at x-coord)
            start = np.argmax(~np.isnan(pos[..., 0]), axis=-1, keepdims=True)

            # calculate the time where people leave the system:
            stop = n_timesteps - 1 - np.argmax(~np.isnan(np.flip(pos[:, :, 0], axis=1)), axis=1, keepdims=True)

            # for each pedestrian, calculate distance (as the crow flies) traveled
            pos_start = np.take_along_axis(pos, start[..., np.newaxis], axis=1)[:, 0]
            pos_stop = np.take_along_axis(pos, stop[..., np.newaxis], axis=1)[:, 0]
            d = np.linalg.norm(pos_stop - pos_start, axis=-1)

            # set illegal routes to np.inf
            bools = line_polygon_overlap(np.stack((pos_start, pos_stop), axis=1), cafe_corners)
            d_temp = np.copy(d)
            d_temp[bools] = np.inf

            # Calculate 1-hop distance matrix: pedestrian start point, ped. end point, cafe corners
            n_peds2 = len(pos_start)
            d1 = np.zeros((n_peds2, 6, 6))
            d1[..., [0, 1], [1, 0]] = d_temp.reshape(-1, 1)
            d1[..., 2:, 2:] = cafe_dists

            # distance from start point to cafe corner nodes
            start_cafe_dist = np.linalg.norm(pos_start[:, np.newaxis] - cafe_corners, axis=-1)
            # distance from end point to cafe corner nodes
            stop_cafe_dist = np.linalg.norm(pos_stop[:, np.newaxis] - cafe_corners, axis=-1)

            # set illegal routes to np.inf
            lines = np.stack(np.broadcast_arrays(pos_start[:, np.newaxis], cafe_corners), axis=2)
            bools = line_polygon_overlap(lines, cafe_corners)
            start_cafe_dist[bools] = np.inf
            lines = np.stack(np.broadcast_arrays(pos_stop[:, np.newaxis], cafe_corners), axis=2)
            bools = line_polygon_overlap(lines, cafe_corners)
            stop_cafe_dist[bools] = np.inf
            d1[..., 0, 2:] = start_cafe_dist
            d1[..., 2:, 0] = start_cafe_dist
            d1[..., 1, 2:] = stop_cafe_dist
            d1[..., 2:, 1] = stop_cafe_dist

            # 1-hop distance from start to end
            d1_m = d1[..., 0, 1]

            # 2-hop distance from start to end
            d2 = d1[..., np.newaxis] + d1[:, np.newaxis]
            d2_m = np.min(d2[:, 0, :, 1], axis=-1)

            # 3-hop distance from start to end
            d3 = d2[..., np.newaxis] + d1[:, np.newaxis, np.newaxis]
            d3_m = np.min(d3[:, 0, :, :, 1], axis=(1, 2))

            d_min = np.min((d1_m, d2_m, d3_m), axis=0)

            # for each pedestrian, calculate arc length traveled
            temp = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
            al = np.nansum(temp, axis=1)

            # print(d, d_min)

            # tortuosity
            tort_this = al / d_min
            tort.extend(tort_this)

            # tortuosity
            tort_this_2 = al / d
            tort2.extend(tort_this_2)

    plt.figure()
    t_gt = np.array(tort_gt)
    t_samp = np.array(tort_samp)
    all_dens_gt = np.array(all_dens_gt)
    all_dens_samp = np.array(all_dens_samp)

    lims = [max(min(np.nanmin(t_gt), np.nanmin(t_samp)), 1),
                max(np.nanquantile(t_gt, 0.99), np.nanquantile(t_samp, 0.99))
                ]

    for yscale in ['log']:

        fig, ax1 = plt.subplots(figsize=set_size_plots(
            width=397, fraction=1 / 2.25 * 9 / 10 * 6 / 5,
            h_to_w_ratio=(5 ** .5 - 1) / 2 * 10 / 9 * 5 / 6))
        fname = f"pedestrian_tortuosity/{cfg['experiment']['state_dict_fname'][:-3]}"
        fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
        if not os.path.exists(fname):
            os.makedirs(fname)

        plot_density_single_plot_v2(t_gt, None, bins_gt=np.linspace(*lims, 25),
                            bins_sampled=np.linspace(*lims, 25), ax=ax1, s=32)
        colors = ['tab:blue', 'tab:purple']
        linestyles = ['-', '--']
        labels = ['Simulated ($T^*$)', 'Simulated ($T^0$)']
        for i in range(len(data_sampled_list)):
            plot_density_single_plot_v2(None, t_samp[i], bins_gt=np.linspace(*lims, 25),
                                        bins_sampled=np.linspace(*lims, 25), ax=ax1, s=32, color_sampled=colors[i],
                                        linestyle_sampled=linestyles[i], sampled_label=labels[i])

        #### inset plot: autocorrelation:
        if autocorr_inset is not None:
            corr_gt = calculate_auto_corr(data_gt, feat_idx=autocorr_inset, max_lag=25)
            left, bottom, width, height = [0.625, 0.66, 0.35, 0.29]  # [0.575, 0.66, 0.4, 0.3]
            ax_inset = fig.add_axes([left, bottom, width, height])
            for i in range(len(data_sampled_list)):
                corr_sample_i = calculate_auto_corr(data_sampled_list[i], feat_idx=autocorr_inset, max_lag=25)
                ax_inset.plot(range(len(corr_sample_i)), corr_sample_i, linestyles[i], color=colors[i],
                              markerfacecolor='none',
                              zorder=-1,
                              linewidth=2)
            ax_inset.scatter(range(len(corr_gt))[::2], corr_gt[::2], marker=r'$\odot$', color='tab:red',
                             facecolor='none', zorder=1, s=32)
            ax_inset.set_xlabel(r'$\tau$ [s]')
            subscr = ['x', 'y', 'u', 'v'][autocorr_inset]
            ax_inset.set_ylabel(rf"$C_{subscr}(\tau)$")
        ax1.set_yscale(yscale)
        ax1.set_xlabel('Tortuosity')
        ax1.set_ylabel('PDF')
        ax1.legend(loc='lower left', ncols=3, frameon=False, columnspacing=0.1,
                   bbox_to_anchor=(-0.025, -0.075))
        ax1.set_ylim(bottom=1e-2)
        # ax1.set_xlim(right=1.6)

        # plot legend fig 2/save legend fig 2:
        legend = ax1.get_legend()
        bbox = legend.get_window_extent()
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        fig_legend = plt.figure(figsize=(width, height))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        fig_legend.legend(handles=[legend.legendHandles[i] for i in (0,1,2)], labels=[
            t.get_text() for t in [legend.get_texts()[i] for i in (0,1,2)]
        ], loc='center',  ncols=3, frameon=False)
        fig_legend.savefig(os.path.join(fname, f"legend" + exten), bbox_inches='tight', transparent=True)
        plt.show()

        ax1.get_legend().remove()
        fig.savefig(os.path.join(fname, f"tortuosity_{yscale}" + exten), bbox_inches='tight', transparent=True)
        print('done with tort/autocorr')


    print('Tortuosity done')


def force_scaling_neighbor_directions(cfg, delta_x_range=(-5000, 5000), vy=-50, num_repetitions=10,
                                      prepend_to_title='', append_to_title='', num_x=12, num_y=25,
                                      vel_mults=(1, -3), cmap='rainbow', delta_y_range=None,
                                      xrange=None, yrange=None, crowd_range=(1,2),
                                      get_crowd_as_lattice=False,
                                      crowd_std=0,
                                      crowd_lattice_width=2000, kind='same'):
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(
        cfg,
        True,
        get_only_model=True
    )


    device = cfg['device']
    model.to(device)

    data_auxiliary = get_auxiliary_data(cfg['use_initial_pos_as_feature']).to(device)  # nodes, feat

    # output objects for saving experiment results:
    all_accs = []
    all_baseline_accs = []
    all_crowd_sizes = []
    all_xs = []
    all_dxs = []
    all_dys = []
    all_vel_mults = []
    all_ys = []
    all_vys = []

    # do the experiment
    for _ in tqdm(range(num_repetitions)):
        vel_mult = np.random.choice(vel_mults)
        # sample lateral distances randomly
        dx = np.random.uniform(low=delta_x_range[0], high=delta_x_range[1], size=num_x * num_y)
        if delta_y_range is None:
            dy = 0 if vel_mult == 1 else -3000  # tangential distance
        else:
            dy = np.random.uniform(low=delta_y_range[0], high=delta_y_range[1], size=num_x * num_y)


        (XX, YY), acc, crowd_size, dx_used, dy_used, mask_used = force_scaling_once_for_all_loc(cfg, model, data_auxiliary, crowd_range,
                                                                                     crowd_com_dist_x=dx,
                                                                                     crowd_com_dist_y=dy,
                                                                                     crowd_std=crowd_std, x_vel=0,
                                                                                     y_vel=vy * np.abs(vel_mult), # both at the same speed, possibly opposite directions
                                                                                     x_vel_crowd=0,
                                                                                     y_vel_crowd=vy * vel_mult,
                                                                                     num_x=num_x, num_y=num_y,
                                                                                     xrange=xrange, yrange=yrange,
                                                                                     get_crowd_as_lattice=get_crowd_as_lattice,
                                                                                     crowd_lattice_width=crowd_lattice_width,
                                                                                     )
        # save output
        all_accs.append(acc)
        all_crowd_sizes.append(crowd_size)
        all_xs.append(XX)
        all_ys.append(YY)
        all_dxs.append(dx_used)
        all_dys.append(dy_used)
        all_vel_mults.append(torch.ones_like(XX) * vel_mult)
        all_vys.append(torch.ones_like(acc[:, 0]) * vy)
        baseline_acc, _ = calc_acc_for_locs(cfg, model, XX, YY, data_auxiliary, 0, vy)
        all_baseline_accs.append(baseline_acc)

    all_accs = torch.cat(all_accs, dim=0).numpy()
    all_baseline_accs = torch.cat(all_baseline_accs, dim=0).numpy()
    all_crowd_sizes = torch.cat(all_crowd_sizes, dim=0).numpy()
    all_xs = torch.cat(all_xs, dim=0).numpy()
    all_ys = torch.cat(all_ys, dim=0).numpy()
    all_dxs = torch.cat(all_dxs, dim=0).numpy()
    all_dys = torch.cat(all_dys, dim=0).numpy()
    all_vel_mults = torch.cat(all_vel_mults, dim=0).numpy()
    all_vys = torch.cat(all_vys, dim=0).numpy()
    all_vxs = np.zeros_like(all_vys)

    # prepare results for plot:
    df = pd.DataFrame({'acc_x': all_accs[:, 0] / 100, 'acc_y': all_accs[:, 1] / 100, 'crowd_size': all_crowd_sizes,
                       'vx': all_vxs / 100, 'vy': all_vys / 100, 'x': all_xs / 1000, 'y': all_ys / 1000,
                       'dx': all_dxs / 1000, 'dy': all_dys / 1000,
                        'vel_mult': all_vel_mults,
                       'baseline_acc_x': all_baseline_accs[:, 0] / 100,
                       'baseline_acc_y': all_baseline_accs[:, 1] / 100})
    n_bins = 50
    n_bins_large = 25

    df['acc_x_corr'] = (df['acc_x'] - df['baseline_acc_x'])
    df['acc_y_corr'] = (df['acc_y'] - df['baseline_acc_y'])
    df['baseline_acc_x_corr'] = 0
    df['baseline_acc_y_corr'] = 0
    df['acc_norm'] = np.linalg.norm(df[['acc_x', 'acc_y']].values, axis=1)
    df['baseline_acc_norm'] = np.linalg.norm(df[['baseline_acc_x', 'baseline_acc_y']].values, axis=1)
    df['acc_norm_corr'] = np.linalg.norm(df[['acc_x_corr', 'acc_y_corr']].values, axis=1)
    df['baseline_acc_norm_corr'] = 0
    df['acc_x_corr_norm'] = df['acc_x_corr'] * np.sign(df['dx'])
    df['abs_dx'] = np.abs(df['dx'])
    df['dist'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

    df['dx_bin'] = pd.cut(df['dx'], bins=n_bins)
    df['dy_bin'] = pd.cut(df['dy'], bins=n_bins)
    df['dx_bin_norm'] = pd.cut(df['abs_dx'], bins=n_bins)
    df['dist_bin'] = pd.cut(df['dist'], bins=n_bins)

    df['dx_bin_large'] = pd.cut(df['dx'], bins=n_bins_large)
    df['dy_bin_large'] = pd.cut(df['dy'], bins=n_bins_large)
    df['dx_bin_large_norm'] = pd.cut(df['abs_dx'], bins=n_bins_large)
    df['dist_bin_large'] = pd.cut(df['dist'], bins=n_bins_large)

    df['acc_inner_prod'] = (df['acc_x_corr'] * df['dx'] + df['acc_y_corr'] * df['dy']) / df['dist']


    fname = f"neighbor_scaling_vel_dir/{cfg['experiment']['state_dict_fname'][:-3]}"
    fname = os.path.join('figures/', prepend_to_title, fname, append_to_title)
    if not os.path.exists(fname):
        os.makedirs(fname)

    # start making the plot
    from matplotlib.colors import TwoSlopeNorm
    cmap = matplotlib.colormaps['bwr']

    if kind == 'same':  # if we are walking in the same direction -- fit ellipsoid
        def ellipsoid_eq(coords, a, b):
            x, y = coords[:, 0], coords[:, 1]
            return (x / a) ** 2 + (y / b) ** 2 - 1

        ############ main plot:

        # crunch the numbers -- group by relative position:
        df_temp = df.copy()
        df_temp = df_temp[(df_temp['vel_mult'] == 1)]
        gb = df_temp[['dx_bin_large', 'dy_bin_large', 'acc_x', 'acc_y', 'acc_x_corr', 'acc_x_corr_norm', 'acc_y_corr',
                      'acc_norm_corr', 'acc_inner_prod']].groupby(['dx_bin_large', 'dy_bin_large']).agg(
            ['mean', 'std'])
        gb['dx'] = pd.IntervalIndex(gb.index.get_level_values(0)).mid
        gb['dy'] = pd.IntervalIndex(gb.index.get_level_values(1)).mid
        gb['d'] = np.sqrt(gb['dx'] ** 2 + gb['dy'] ** 2)

        num_x, num_y = gb['dx'].nunique(), gb['dy'].nunique()
        dxx, dyy = gb['dx'].values.reshape(num_x, num_y), gb['dy'].values.reshape(num_x, num_y)
        acc_x_corr = gb['acc_x_corr']['mean'].values.reshape(num_x, num_y)
        # calculate relative acceleration, where the sign is determined by repulsive vs attractive force
        acc_x_att_repul = (acc_x_corr - np.flip(acc_x_corr)) * -1 * np.sign(dxx)


        ## start plotting:
        norm = TwoSlopeNorm(vcenter=0, vmin=-np.nanmax(np.abs(acc_x_att_repul)), vmax=np.nanmax(np.abs(acc_x_att_repul)))
        scaling_factor_plotsize = 0.8
        fig, ax = plt.subplots(figsize=(4.8 * scaling_factor_plotsize, 3.6 * scaling_factor_plotsize),
                               layout='constrained')

        # relative lateral acc contour plot:
        plt.contourf(dyy, dxx, acc_x_att_repul, levels=5, cmap=cmap, norm=norm)
        cbar = plt.colorbar()
        cbar.set_label('$[m/s^2]$')
        cbar.ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)  # black line at 0

        # 0 acc contour:
        acc_x_att_repul_center_removed = acc_x_att_repul.copy()  # remove small displacements for contour fitting
        acc_x_att_repul_center_removed[(np.abs(dxx) < 0.25) & (np.abs(dyy) < 3.)] = np.nan
        contour = plt.contour(dyy, dxx, acc_x_att_repul_center_removed, levels=[0], colors='black', linewidths=1,
                              linestyles='--', alpha=0.5)

        # fit an ellipsoid to the 0 contour:
        paths = contour.collections[0].get_paths()
        v = paths[0].vertices  # Extract vertices of the 0-level curve
        coords = v
        popt_ellipsoid, pcov_ellipsoid = curve_fit(ellipsoid_eq, coords, np.zeros_like(coords[:, 0]),
                                                   bounds=([0.0001, 0.0001], [100., 100.]))
        a, b = popt_ellipsoid
        # plot the ellipsoid:
        theta = np.linspace(0, 2 * np.pi, 100)
        y_ell = a * np.cos(theta)
        x_ell = b * np.sin(theta)
        ax.plot(y_ell, x_ell, '-', color='black', label=f'ellipsoid fit', alpha=1, linewidth=2)

        plt.plot([0], [0], 'o', markersize=12, color='black')
        plt.arrow(0, 0, -1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

        ax.set_xlabel(r'$\Delta x$ [m]', fontsize=12)
        ax.set_ylabel(r'$\Delta y$ [m]', fontsize=12)
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        plt.savefig(os.path.join(fname, 'attr_repul_sameflow.svg'), bbox_inches='tight')
        plt.show()

    elif kind == 'opposite':  # neighbor is walking in the opposite direction
        # group by dx, dy and get the mean acc_x_corr, acc_y_corr, acc_norm_corr:
        df_temp = df.copy()
        df_temp = df_temp[(df_temp['vel_mult'] == -1)]  # -1 for opp
        gb = df_temp[['dx_bin', 'dy_bin', 'acc_x', 'acc_y', 'acc_x_corr', 'acc_x_corr_norm', 'acc_y_corr', 'acc_norm_corr',
                      'acc_inner_prod']].groupby(['dx_bin', 'dy_bin']).agg(['mean', 'std'])
        gb['dx'] = pd.IntervalIndex(gb.index.get_level_values(0)).mid
        gb['dy'] = pd.IntervalIndex(gb.index.get_level_values(1)).mid
        gb['d'] = np.sqrt(gb['dx'] ** 2 + gb['dy'] ** 2)


        # make a contour plot of the lateral acceleration:
        num_x, num_y = gb['dx'].nunique(), gb['dy'].nunique()
        dxx, dyy = gb['dx'].values.reshape(num_x, num_y), gb['dy'].values.reshape(num_x, num_y)
        acc_x_corr = gb['acc_x_corr']['mean'].values.reshape(num_x, num_y) * 2  # *2 for relative acc -- point symmetric dynamics assumption
        acc_x_att_repul = acc_x_corr * -1 * np.sign(dxx)  # sign indicates repulsive/attractive

        # plot the contour plot:
        cmap = matplotlib.colormaps['bwr']
        norm = TwoSlopeNorm(vcenter=0, vmin=-np.nanmax(np.abs(acc_x_att_repul)), vmax=np.nanmax(np.abs(acc_x_att_repul)))
        scaling_factor_plotsize = 0.8
        fig, ax = plt.subplots(figsize=(4.8 * scaling_factor_plotsize, 3.6 * scaling_factor_plotsize), layout='constrained')
        ax.grid(False)
        plt.contourf(dyy, dxx, acc_x_att_repul, levels=5, cmap=cmap, norm=norm)
        cbar = plt.colorbar()
        cbar.set_label('$[m/s^2]$')
        plt.plot([0], [0], 'o', markersize=12, color='black')
        plt.arrow(0, 0, -1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

        # plot some lines according to the model proposed in Corbetta (2018):
        def plot_field_of_view(angle, num_points=5, **kwargs):
            x_start, y_start = 0, 0
            x_end, y_end = -4 * np.tan(np.deg2rad(angle)), -4
            xrange = np.linspace(x_start, x_end, num_points)
            yrange = np.linspace(y_start, y_end, num_points)
            plt.plot(yrange, xrange, **kwargs)
            if 'label' in kwargs:
                kwargs.pop('label')
            plt.plot(yrange, -xrange, **kwargs)


        plot_field_of_view(20, color='black', linestyle=':', linewidth=2,
                           label=r'$\theta =20^\circ$ (Corbetta, 2018)')  # Corbetta's "vision force" angle in degrees
        degree_us = np.rad2deg(np.arctan(2 / 4))  # visual approximate fit
        plot_field_of_view(degree_us, color='black', linestyle='--', linewidth=2,
                           label=r'$\theta \approx$' + str(float(np.round(degree_us, 1))) + r'$^\circ$')
        ax.set_xlim(-3, 0.1)
        ax.set_ylim(-3., 3.)

        ######## finalizing:
        # make the background of the legend a buit more transparent
        ax.legend(loc='lower center', framealpha=0.5)
        ax.set_xlabel(r'$\Delta x$ [m]', fontsize=12)
        ax.set_ylabel(r'$\Delta y$ [m]', fontsize=12)
        plt.savefig(os.path.join(fname, 'attr_repul_oppflow.svg'), bbox_inches='tight')
        plt.show()

    print('plots complete')


@torch.no_grad()
def counterflow_transfer_function(cfg, delta_x_range=(-2500, 2500), delta_ys=(-8000,), y_vels=(-50,),
                                  num_repetitions=30, num_x=10, num_y=10, tmax=15,
                                  prepend_to_title='', append_to_title='', dataframes=None, corbetta_data=None,
                                  xrange_start=(-8500, 3900), yrange_start=(10000, 75000)):


    if dataframes is None:  # generate simulated data


        model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(
            cfg,
            True,
            get_only_model=True
        )

        device = cfg['device']
        model.to(device)
        data_auxiliary = get_auxiliary_data(cfg['use_initial_pos_as_feature']).to(device)  # nodes, feat

        all_delta_x_t0_counterflow = []
        all_delta_x_t1_counterflow = []
        all_t1 = []
        xrange = (xrange_start[0] - delta_x_range[0], xrange_start[1] - delta_x_range[1])

        # iterate over experimental conditions:
        all_conditions = list(itertools.product(delta_ys, y_vels))
        for delta_y, y_vel in tqdm(all_conditions * num_repetitions):

            yrange = (yrange_start[0] - delta_y, yrange_start[1] - delta_y)
            # make a batch of samples for each condition, over all locations:
            dx = np.random.uniform(low=delta_x_range[0], high=delta_x_range[1], size=num_x * num_y)

            ####  counterflow:

            all_test_data, all_batch_idx, all_crowd_sizes, all_dx, all_dy, x_used, y_used, mask_used = get_synth_setup_crowd_for_all_loc(
                cfg, data_auxiliary, crowd_range=(1, 2), crowd_com_dist_x=dx, crowd_com_dist_y=delta_y, crowd_std=0,
                x_vel=0,
                y_vel=y_vel, x_vel_crowd=0, y_vel_crowd=-y_vel, num_x=num_x, num_y=num_y, xrange=xrange, yrange=yrange
            )  # this returns the data in model internal coordinates, but the other outputs in real coordinates
            all_delta_x_t0_counterflow.append(all_dx)

            all_test_data = all_test_data[:, :, None, :].repeat(1, 1, tmax, 1).to(device)  # (nodes, 1, tmax, feat)
            # for the non-flowing pedestrian, we need to initialize their positions with some values for the model rollout function
            # later these will be overwritten by the symmetry-assuming rollout. Lets assume constant vel for the dummy values.
            nonflowing_pedes_mask = torch.bitwise_and(all_test_data[:, 0, 0, 7] == 0, all_test_data[:, 0, 0, 4] == 1)
            # we can assume standard normalization:
            all_ts = torch.arange(tmax, device=device).float()  # t in seconds, pos in decameter, vel in m/s for model internal coors
            all_test_data[nonflowing_pedes_mask, 0, :, 1] = all_test_data[nonflowing_pedes_mask, 0, 0:1,
                                                            1] + all_ts * 0.1 * all_test_data[nonflowing_pedes_mask, 0, 0:1,
                                                                                3]  # linear extrapolation, will be overwritten

            # now run the model assuming point-symmetrical accelerations:
            trues, preds, _ = model_rollout_assume_symmetry(model, all_test_data, 1, tmax - 1, start_time=0,
                                            use_posterior_sampling=False, use_ground_truth_input=False,
                                            miscellaneous_all=[all_batch_idx], run_with_error_handling=False, config=cfg)

            preds_samples = np.concatenate([i[:, :, None, ...] for i in preds], axis=2)  # (people, 1, time, feat)
            trues = np.concatenate([i[:, :, None, ...] for i in trues], axis=2)  # (people, 1, time, feat)

            preds_samples[..., :4] = utils.to_real_coordinates(preds_samples[..., :4], True, False)
            trues[..., :4] = utils.to_real_coordinates(trues[..., :4], True, False)
            all_batch_idx = all_batch_idx.cpu().numpy()

            # now, for each batch_idx, find the time where the difference in the y-coordinate of the two people is minimal:
            t1 = []
            dx_list = []
            for b in np.unique(all_batch_idx):
                ys_this_batch = preds_samples[all_batch_idx == b, 0, :, 1][
                    preds_samples[all_batch_idx == b, 0, 0, 4] == 1]  # (people, time)
                xs_this_batch = preds_samples[all_batch_idx == b, 0, :, 0][
                    preds_samples[all_batch_idx == b, 0, 0, 4] == 1]  # (people, time)
                assert ys_this_batch.shape[0] == 2, 'there should be two people in this batch'
                delta_y_simulated = ys_this_batch[1] - ys_this_batch[0]  # (time)
                delta_x_simulated = xs_this_batch[1] - xs_this_batch[0]  # (time)
                t_min = np.argmin(np.abs(delta_y_simulated))
                dx_list.append(delta_x_simulated[t_min])
                t1.append(t_min)


            t1 = np.stack(t1, axis=0)  # (batch_idx,)
            dx_list_cat = np.stack(dx_list, axis=0)  # (batch_idx,)
            all_delta_x_t1_counterflow.append(dx_list_cat)
            all_t1.append(t1)



        all_delta_x_t0_counterflow = np.concatenate(all_delta_x_t0_counterflow, axis=0)
        all_delta_x_t1_counterflow = np.concatenate(all_delta_x_t1_counterflow, axis=0)
        all_t1 = np.concatenate(all_t1, axis=0)

        # prepare the plot:
        df_counter = pd.DataFrame({'delta_x_t0_counterflow': all_delta_x_t0_counterflow,
                                   'delta_x_t1_counterflow': all_delta_x_t1_counterflow})
        df_counter['delta_x_t0_abs'] = np.abs(df_counter['delta_x_t0_counterflow'])
        df_counter['delta_x_t1_abs'] = np.abs(df_counter['delta_x_t1_counterflow'])

        fname = f"counterflow_transfer_function/{cfg['experiment']['state_dict_fname'][:-3]}"
        fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
        if not os.path.exists(fname):
            os.makedirs(fname)
        # save the dfs:
        df_counter.to_csv(os.path.join(fname, 'df_counter.csv'))
        # df_same.to_csv(os.path.join(fname, 'df_same_xy.csv'))
    else:
        path_counter, *misc = dataframes
        df_counter = pd.read_csv(path_counter)
        # df_same = pd.read_csv(path_same)
        print('loaded dfs from disk!')

    def lower(x):
        return np.mean(x) - np.std(x)
        # return np.quantile(x, 0.1)

    def upper(x):
        return np.mean(x) + np.std(x)

    # convert from mm to m:
    df_counter = df_counter[['delta_x_t0_abs', 'delta_x_t1_abs']] / 1000.
    bin_counter_t0 = pd.cut(df_counter['delta_x_t0_abs'], bins=20)

    fname = f"counterflow_transfer_function/{cfg['experiment']['state_dict_fname'][:-3]}"
    fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
    if not os.path.exists(fname):
        os.makedirs(fname)

    mean_dx_counter = df_counter.groupby(bin_counter_t0).agg(['mean', lower, upper])

    dx_abs_max = max(np.abs(delta_x_range[0]), np.abs(delta_x_range[1]))

    # plot counterflow:
    fig = plt.figure(figsize=set_size_plots(width, fraction=0.857 / 6 * 2.5, h_to_w_ratio=1))

    plt.plot(mean_dx_counter['delta_x_t0_abs']['mean'], mean_dx_counter['delta_x_t1_abs']['mean'], '-',
             color='tab:blue',
             label='Simulated',
             linewidth=2, zorder=10)
    # if we have corbetta 2018 data, plot it:
    if corbetta_data is not None:
        corbetta: pd.DataFrame = pd.read_hdf(corbetta_data, 'edge', complevel=9, complib='blosc')
        corbetta = corbetta.abs()
        bin_corbetta_t0 = pd.cut(corbetta['dyInitial'], bins=25)
        mean_dx_corbetta = corbetta.groupby(bin_corbetta_t0).agg(['mean', lower, upper])
        plt.plot(mean_dx_corbetta['dyInitial']['mean'], mean_dx_corbetta['dyCenter']['mean'], ':',
                 color='black',
                 label='Corbetta (2018)', linewidth=2, zorder=10)

        # indices for error bars in Corbetta data
        indices_errorbar = [1, 5, 8]
        plt.errorbar(
            mean_dx_corbetta['dyInitial']['mean'].iloc[indices_errorbar],
            mean_dx_corbetta['dyCenter']['mean'].iloc[indices_errorbar],
            yerr=[
                mean_dx_corbetta['dyCenter']['mean'].iloc[indices_errorbar] -
                mean_dx_corbetta['dyCenter']['lower'].iloc[indices_errorbar],
                mean_dx_corbetta['dyCenter']['upper'].iloc[indices_errorbar] -
                mean_dx_corbetta['dyCenter']['mean'].iloc[indices_errorbar],
            ],
            fmt='none',
            linewidth=1,
            color='black',
            capsize=3,
            zorder=11
        )
    plt.plot([0, dx_abs_max / 1000], [0, dx_abs_max / 1000], '-.', color='black', alpha=0.5, zorder=-10)
    plt.xlabel(r'$|\Delta y_i |$ [m]')
    plt.ylabel(r'$|\Delta y_s |$ [m]')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    legend = fig.legend(frameon=False, loc='outside upper center', ncols=2)

    # # plot legend:
    bbox = legend.get_window_extent()
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    width_legend, height_legend = bbox.width, bbox.height
    fig_legend = plt.figure(figsize=(width_legend, height_legend))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    fig_legend.legend(handles=legend.legendHandles, labels=[
        t.get_text() for t in legend.get_texts()], loc='center', ncols=1, frameon=False)
    fig_legend.savefig(os.path.join(fname, f"legend.svg"), bbox_inches='tight', transparent=True)
    # plt.show()

    legend.remove()

    fig.savefig(os.path.join(fname, f"counterflow_transfer_function.svg"), bbox_inches='tight',
                transparent=True)
    plt.show()

    print('all done!')




def force_scaling_crowd(cfg, num_repetitions=10, x_vel_range=(0, 0), y_vel_range=(-200, 0), crowd_range=(1, 25),
                        crowd_com_dist_range=(-5000, -1000), crowd_std=500,
                        num_x=12, num_y=25, num_x_vel=1, num_y_vel=20, get_crowd_as_lattice=False,
                        crowd_lattice_width=2000, y_vel_crowd_range=(0.,0.000001),
                        top_ks_social_forcing=(3, 9, 12, 15, 20),
                        cmap='rainbow', prepend_to_title='',
                        append_to_title='', exten='.svg', load_csv_path=None,
                        pre_loaded_model=None,
                        yrange=None,
                        U0=0.375, R=2., max_angle_sf=2*np.pi,
                        add_spacing_center_lattice_range=(0., 0.00001)):

    figsize['figure.figsize'] = set_size_plots(width=397,
                                               fraction=1 / 2.25 * 9 / 10, h_to_w_ratio=(5 ** .5 - 1) / 2 * 10 / 9)

    plt.rcParams.update(figsize)
    fname = f"force_scaling/{cfg['experiment']['state_dict_fname'][:-3]}"
    fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
    if not os.path.exists(fname):
        os.makedirs(fname)

    if load_csv_path is None: # re-generate experiment measurements
        # model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(
        #     cfg,
        #     True,
        #     get_only_model=True
        # )

        if pre_loaded_model is None:
            model, *_ = load_model_and_get_dataloaders(
                cfg,
                True,
                get_only_model=True
            )
        else:
            model = pre_loaded_model

        device = cfg['device']
        model.to(device)

        data_auxiliary = get_auxiliary_data(cfg['use_initial_pos_as_feature']).to(device)  # nodes, feat

        # now generate the velocities to test
        vel_combs = list(itertools.product(
            np.linspace(*x_vel_range, num_x_vel), np.linspace(*y_vel_range, num_y_vel)
        ))

        # initialize objects to store experimental results:
        all_accs = []
        all_social_accs = []
        all_topk_social_accs = {k: [] for k in top_ks_social_forcing}
        all_baseline_accs = []
        all_crowd_sizes = []
        all_xs = []
        all_ys = []
        all_vxs = []
        all_vys = []
        all_dys = []
        all_lattice_spacings = []

        ## get a full lattice for vis purposes:
        # range_max = (crowd_range[-1] - 1, crowd_range[-1])
        # example_test_data, example_batch_idx, *_ = get_synth_setup_crowd_for_all_loc(
        #     cfg, data_auxiliary, range_max, 0, crowd_com_dist_range[0], crowd_std, 0, 0,
        #     0, 0, num_x, num_y, get_crowd_as_lattice=get_crowd_as_lattice,
        #     crowd_lattice_width=crowd_lattice_width,
        #     add_spacing_center_lattice=add_spacing_center_lattice_range[1]
        # )
        # smallest_succesful_batch_idx = torch.min(example_batch_idx)
        # example_test_data = example_test_data[example_batch_idx == smallest_succesful_batch_idx].cpu().numpy()
        # example_test_data = example_test_data[example_test_data[:, 0, 4] == 1]
        # # plt.scatter(example_test_data[:, 0, 1]*10, example_test_data[:, 0, 0]*10)
        # # plt.savefig(os.path.join(fname, f"example_lattice.pdf"), bbox_inches='tight')

        for vx, vy in tqdm(vel_combs * num_repetitions):  # vel_combs is a list of tuples (vx, vy)
            # get random crowd parameters from the prespecified ranges:
            dy = np.random.uniform(crowd_com_dist_range[0], crowd_com_dist_range[1], size=num_x * num_y)  # displacement
            y_vel_crowd = np.random.uniform(y_vel_crowd_range[0], y_vel_crowd_range[1], size=num_x*num_y)  # crowd vel
            add_spacing_center_lattice = np.random.uniform(*add_spacing_center_lattice_range, size=num_x*num_y) # additional spacing between center lines of crowd

            # calculate the acceleration on the ego-pedestrian for all locations/crowd params:
            (XX, YY), acc, crowd_size, all_dx, all_dy, social_accs, top_k_social_accs, mask_used = force_scaling_once_for_all_loc(
                cfg, model, data_auxiliary, crowd_range,
                crowd_com_dist_y=dy,
                crowd_std=crowd_std, # lattice spacing for lattice crowd, coord std for Gaussian
                x_vel=vx, y_vel=vy,  # vel of ego-pedestrian
                x_vel_crowd=0, y_vel_crowd=y_vel_crowd,  # vel of crowd
                num_x=num_x, num_y=num_y, #  x \times y locations to conduct the experiment at
                get_crowd_as_lattice=get_crowd_as_lattice, # whether to get a lattice configuration or gaussian crowd
                crowd_lattice_width=crowd_lattice_width,  # total width of the lattice in case of the lattice config
                calculate_social_forcing=True,  # whether to also calculate social force
                top_ks_social_forcing=top_ks_social_forcing,  # for which k values to calculate the top-k sf
                yrange=yrange,  # y-coordinate range to conduct the experiments
                U0=U0, R=R, max_angle_sf=max_angle_sf,  # social force parameters
                add_spacing_center_lattice=add_spacing_center_lattice  # additional spacing between center lines of crowd (lattice case)
            )

            all_accs.append(acc)
            all_social_accs.append(social_accs)
            all_crowd_sizes.append(crowd_size)
            all_xs.append(XX)
            all_ys.append(YY)
            all_vxs.append(torch.ones_like(acc[:, 0]) * vx)
            all_vys.append(torch.ones_like(acc[:, 0]) * vy)
            all_dys.append(torch.ones_like(acc[:, 0]) * all_dy)
            all_lattice_spacings.append(add_spacing_center_lattice[mask_used])
            for k in all_topk_social_accs.keys():
                all_topk_social_accs[k].append(top_k_social_accs[k])

            # also calculate the baseline acceleration in the presence of no neighbors to discount this from the force
            baseline_acc, _ = calc_acc_for_locs(cfg, model, XX, YY, data_auxiliary, vx, vy)
            all_baseline_accs.append(baseline_acc)


        all_accs = torch.cat(all_accs, dim=0).numpy()
        all_social_accs = torch.cat(all_social_accs, dim=0).numpy()
        all_baseline_accs = torch.cat(all_baseline_accs, dim=0).numpy()
        all_crowd_sizes = torch.cat(all_crowd_sizes, dim=0).numpy()
        all_xs = torch.cat(all_xs, dim=0).numpy()
        all_ys = torch.cat(all_ys, dim=0).numpy()
        all_vxs = torch.cat(all_vxs, dim=0).numpy()
        all_vys = torch.cat(all_vys, dim=0).numpy()
        all_dys = torch.cat(all_dys, dim=0).numpy()
        all_topk_social_accs = {k: torch.cat(v, dim=0).numpy() for k, v in all_topk_social_accs.items()}
        all_lattice_spacings = np.concatenate(all_lattice_spacings, axis=0)

        df = pd.DataFrame({'acc_x': all_accs[:, 0], 'acc_y': all_accs[:, 1], 'social_acc_x': all_social_accs[:, 0],
                           'social_acc_y': all_social_accs[:, 1],
                           'crowd_size': all_crowd_sizes,
                           'vx': all_vxs, 'vy': all_vys, 'x': all_xs, 'y': all_ys,
                           'baseline_acc_x': all_baseline_accs[:, 0],
                           'baseline_acc_y': all_baseline_accs[:, 1],
                           'dy': all_dys,
                          'lattice_spacing': all_lattice_spacings})
        df['social_acc_norm'] = np.linalg.norm(df[['social_acc_x', 'social_acc_y']].values, axis=1)
        for k, v in all_topk_social_accs.items():
            df[f'top{k}_social_acc_x'] = v[:, 0]
            df[f'top{k}_social_acc_y'] = v[:, 1]
            df[f'top{k}_social_acc_norm'] = np.linalg.norm(v, axis=1)

        # correct acc for no-neighbor acc to get the sf
        df['acc_x_corr'] = df['acc_x'] - df['baseline_acc_x']
        df['acc_y_corr'] = df['acc_y'] - df['baseline_acc_y']
        df['baseline_acc_x_corr'] = 0
        df['baseline_acc_y_corr'] = 0
        df['acc_norm'] = np.linalg.norm(df[['acc_x', 'acc_y']].values, axis=1)
        df['baseline_acc_norm'] = np.linalg.norm(df[['baseline_acc_x', 'baseline_acc_y']].values, axis=1)
        df['acc_norm_corr'] = np.linalg.norm(df[['acc_x_corr', 'acc_y_corr']].values, axis=1)
        df['baseline_acc_norm_corr'] = 0

        # get current date in YYYY-MM-DD to save csv with results and prevent waiting a long time again:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d")
        df.to_csv(os.path.join(fname, f"{dt_string}_force_scaling_U0_{U0}_R_{R}.csv"), index=False)

    else: # load a csv with earlier results from disk
        df = pd.read_csv(load_csv_path)

    # plot the results -- scaling as a function of crowd size:
    fig, (ax2) = plt.subplots(1, 1, )
    # overall aggregated plot, comparing with social forcing:
    grouped_on_crowd = df.groupby('crowd_size').agg(['mean', 'std'])
    crowd = grouped_on_crowd.index
    ax2.plot(crowd, grouped_on_crowd['acc_norm_corr']['mean'] / 100, '-', color='tab:blue',
             label='Simulated', zorder=10, linewidth=2.5)
    ax2.plot(crowd, grouped_on_crowd['social_acc_norm']['mean'] / 100, '--s', color='tab:orange',
             label='Social force', markerfacecolor='none', zorder=-10, markevery=4, markersize=8)
    for k in [12]:
        ax2.plot(crowd, grouped_on_crowd[f'top{k}_social_acc_norm']['mean'] / 100, '--o', color='tab:green',
                 label=f'Top-k social force', markerfacecolor='none', zorder=-10, markevery=4, markersize=8)
    plt.xlabel('Crowd size')
    plt.xlim(0, min(crowd_range[1] - 1, max(crowd)))
    plt.ylim(0., 0.9)
    plt.vlines(12, 0, 0.9, color='black', linestyle='--', linewidth=1)

    plt.ylabel('Acceleration [$m/s^2$]')
    if exten == '.svg':  # paper
        ax2.legend(frameon=False,)

        # save the legend in a separate file:
        legend = ax2.get_legend()
        bbox = legend.get_window_extent()
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        fig_legend = plt.figure(figsize=(width, height))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        fig_legend.legend(handles=[legend.legendHandles[i] for i in (1,2)], labels=[
            t.get_text() for t in [legend.get_texts()[i] for i in (1,2)]   # only keep sf legend components
        ], loc='center', ncols=2, frameon=False)
        fig_legend.savefig(os.path.join(fname, f"legend" + exten), bbox_inches='tight', transparent=True)
        fig_legend.show()

        ax2.get_legend().remove()
    plt.savefig(os.path.join(fname, f"latt_{get_crowd_as_lattice}_U0_{float(np.round(U0, 2))}_R_{R}" + exten),
                bbox_inches='tight', transparent=True)

    plt.show()

    print('done with force scaling in crowd size')

    # now make a plot grouped on the center spacing:
    fig, ax_spacing = plt.subplots(1, 1, figsize=set_size_plots(397.48499, fraction=1 / 1.5, h_to_w_ratio=None))
    space_bins = pd.cut(df['lattice_spacing'], bins=20)
    grouped_on_spacing = df.groupby(space_bins).agg(['mean', 'std'])
    spacing = pd.IntervalIndex(grouped_on_spacing.index).mid / 1000
    ax_spacing.plot(spacing, grouped_on_spacing['acc_norm_corr']['mean'] / 100, '-', color='tab:blue',
                    label='Simulated', zorder=10, linewidth=2.5)
    ax_spacing.plot(spacing, grouped_on_spacing['social_acc_norm']['mean'] / 100, '--s', color='tab:orange',
                    label='Social force', markerfacecolor='none', zorder=-10, markevery=4, markersize=8)
    for k in [12]:
        ax_spacing.plot(spacing, grouped_on_spacing[f'top{k}_social_acc_norm']['mean'] / 100, '--o', color='tab:green',
                        label=f'Top-k social force', markerfacecolor='none', zorder=-10, markevery=4, markersize=8)
    plt.xlabel('Center spacing [m]', fontsize=12)
    plt.ylabel('Acceleration[$m/s^2$]', fontsize=12)
    plt.savefig(os.path.join(fname, f"scaling_n_lattice_{get_crowd_as_lattice}_U0_{float(np.round(U0,2))}_R_{R}_spacing_center" + exten),
                bbox_inches='tight', transparent=True)
    plt.show()

    print('plots complete')


def collisions(cfg, data_gt, data_sampled_list, dists, vel_diffs, max_angle, calc_vel_from_pos=True):
    import warnings
    warnings.filterwarnings("ignore")
    sampled_labels = ['Simulated ($T=0.1$)', 'Simulated ($T=0$)', 'Simulated (Add. SF)']

    # calculate the number of collisions in the ground-truth and in the model samples:
    # a collision happens if the following criteria are satisfied:
    # 1) the distance between two pedestrians is smaller than epsilon (to be defined)
    # 2) the norm of the difference in velocity between two pedestrians is larger than delta (to be defined)
    # 3) the pedestrians are moving towards each other
    # -> (angle between relative velocity and relative position is between 135 and 225 degrees)
    # this means the cos must be < -sqrt(2) / 2
    #
    # we calculate this for a range of epsilons and deltas
    # fill in the below in meters. multipliers will convert to mm and cm/s


    epsilons = np.array(dists) * 1000  # in mm
    deltas = np.array(vel_diffs) * 100  # in cm/s

    max_cos = np.cos(np.deg2rad(180 - max_angle))  # max_angle in degrees
    num_dens_bins = 10
    col_gt_arr = np.zeros(shape=(num_dens_bins+1, len(epsilons), len(deltas)))
    col_sample_arr = [np.zeros(shape=(num_dens_bins+1, len(epsilons), len(deltas))) for _ in range(len(data_sampled_list))]

    pos_gt = []
    pos_sampled = [[] for _ in range(len(data_sampled_list))]   # each *_sampled is a list of lists, each list corresponds to one entry in the data_sampled_list
    distances_gt = []
    distances_sampled = [[] for _ in range(len(data_sampled_list))]
    nn_dist_gt = []
    nn_dist_sampled = [[] for _ in range(len(data_sampled_list))]
    timestep_gt = []
    timestep_sampled = [[] for _ in range(len(data_sampled_list))]
    local_dens_gt = []
    local_dens_sampled = [[] for _ in range(len(data_sampled_list))]

    # do all calcs for gt and sampled data
    for data, pos_all, dist_all, timestep_all, nn_dist_all, local_dens_all in zip(
            [data_gt, *data_sampled_list], [pos_gt, *pos_sampled], [distances_gt, *distances_sampled],
            [timestep_gt, *timestep_sampled], [nn_dist_gt, *nn_dist_sampled],
            [local_dens_gt, *local_dens_sampled]
    ):
        for i in tqdm(range(len(data))):
            d = data[i]
            if calc_vel_from_pos:
                vel = np.diff(d[..., :2], axis=2)
                d[..., 1:, 2:4] = vel
                d = d[:, :, 1:]

            ### keep only pedes:
            pedes_mask = np.any(d[:,0,:, 4] == 1, axis=1)
            d = d[pedes_mask]
            pos = d[:,0,:, :2]  #(nodes, time, feat)

            # calculate quantities for collision determinance:
            relpos = pos[:, None, ...] - pos[None, ...]  # (nodes, nodes, time, feat)
            dist = np.linalg.norm(relpos, axis=-1)  # (nodes, nodes, time)

            # # calculate local densitities:
            local_dens = calculate_local_density(np.copy(pos[:, None, :,:]), how='voronoi')  # pos in mm
            local_dens_repeated = np.repeat(local_dens[:, None, :], dist.shape[1], axis=1)
            # calculate quantities for nearest neighbor anlaysis:
            dist_diag_inf = np.copy(dist)
            dist_diag_inf[np.eye(dist_diag_inf.shape[0], dtype=bool)] = np.where(
                 ~np.isnan(dist_diag_inf[np.eye(dist_diag_inf.shape[0], dtype=bool)]), np.inf, np.nan)
            nn_dist = np.nanmin(dist_diag_inf, axis=1)  # distance of the nearest neighbor -- nodes, time

            # some supplementary quantities to group the nn calculation by:
            timesteps = np.tile(np.arange(nn_dist.shape[1]), (nn_dist.shape[0], 1))
            m = ~np.isnan(nn_dist)
            nn_dist = nn_dist[m]  #observations
            nn_dist_all.append(nn_dist)
            timestep_all.append(timesteps[m])

            # get only upper triangular part of matrix (above the diagonal):
            idx_upper = np.triu_indices(pos.shape[0], k=1)
            dists_upper = dist[idx_upper]   # all_pairs, time
            local_dens_repeated = local_dens_repeated[idx_upper]

            # remove NaNs:
            non_nan_mask = ~np.isnan(dists_upper)
            dist_all.append(dists_upper[non_nan_mask])  # all_pairs * time
            local_dens_all.append(local_dens_repeated[non_nan_mask])

            pos = pos.reshape(-1, pos.shape[-1])
            non_nan_mask = ~np.isnan(pos).any(axis=1)
            pos_all.append(pos[non_nan_mask])


    pos_gt = np.concatenate(pos_gt, axis=0)
    pos_sampled = [np.concatenate(pos_sampled[i], axis=0) for i in range(len(data_sampled_list))]
    distances_gt = np.concatenate(distances_gt, axis=0)
    distances_sampled = [np.concatenate(distances_sampled[i], axis=0) for i in range(len(data_sampled_list))]
    local_dens_gt = np.concatenate(local_dens_gt, axis=0)
    local_dens_sampled = [np.concatenate(local_dens_sampled[i], axis=0) for i in range(len(data_sampled_list))]

    dens_bins = pd.cut(local_dens_gt, bins=np.linspace(0, 1.4, num_dens_bins+1))


    for i, epsilon in tqdm(enumerate(epsilons), total=len(epsilons)):
        for j, delta in enumerate(deltas):

            collision_gt = distances_gt <= epsilon

            collision_gt_total = np.sum(collision_gt)
            col_gt_arr[0, i, j] = collision_gt_total / pos_gt.shape[0] # first is collisions for all densities
            for dens_idx, dens_bin in enumerate(dens_bins.categories):
                dens_mask_gt = (local_dens_gt >= dens_bin.left) & (local_dens_gt < dens_bin.right)
                collision_gt_dens = np.logical_and(collision_gt, dens_mask_gt)
                col_gt_arr[dens_idx+1, i, j] = np.sum(collision_gt_dens) / dens_mask_gt.sum()

            for data_sampled_id in range(len(data_sampled_list)):
                collision_sample = distances_sampled[data_sampled_id] <= epsilon
                collision_sample_total = np.sum(collision_sample)
                col_sample_arr[data_sampled_id][0, i, j] = collision_sample_total / pos_sampled[data_sampled_id].shape[0]

                for dens_idx, dens_bin in enumerate(dens_bins.categories):
                    dens_mask_sampled = (local_dens_sampled[data_sampled_id] >= dens_bin.left) & (
                                local_dens_sampled[data_sampled_id] < dens_bin.right)
                    collision_sample_dens = np.logical_and(collision_sample, dens_mask_sampled)
                    col_sample_arr[data_sampled_id][dens_idx+1, i, j] = np.sum(collision_sample_dens) / dens_mask_sampled.sum()



    path = f"figures/pedestrian_collision/{cfg['experiment']['state_dict_fname'][:-3]}"
    if not os.path.exists(path):
        os.makedirs(path)


    fsize = set_size_plots(397, fraction=0.857 / 6 * 2.5, h_to_w_ratio=1)
    plt.figure(figsize=fsize)
    colors = ['tab:blue', 'tab:purple', 'tab:brown']
    lstyles = ['-', '--', '-x']
    for i, eps in enumerate(epsilons):
        if eps == 400:  # we only do eps=400mm for the paper appendix as this seems the most reasonable threshold
            print('collision rate:', col_gt_arr[1:, i, 0])
            plt.scatter(dens_bins.categories.mid, 1 / col_gt_arr[1:, i, 0], color='tab:red', s=48,
                        marker='$\odot$', label='Observed')
            for j in range(len(data_sampled_list)):
                plt.plot(dens_bins.categories.mid, 1 / col_sample_arr[j][1:, i, 0], lstyles[j],
                         color=colors[j], label=sampled_labels[j], linewidth=2, markevery=1, markerfacecolor='none',
                         markersize=7, markeredgewidth=2)
    # plt.legend()
    plt.ylabel('Intercollision time [s/($col$)]')
    plt.xlabel('Density [$m^{-2}$]')
    plt.yscale('log')
    plt.ylim(50 ** 1, 10 ** 4)
    plt.xlim(0, 1.4)
    plt.savefig(os.path.join(path, 'intercollision_time.svg'), bbox_inches='tight', transparent=True)
    plt.show()
    print('done')