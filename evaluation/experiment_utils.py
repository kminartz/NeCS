import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap
from matplotlib import colormaps
from matplotlib import patches
from matplotlib import collections
from utils import filter_input, construct_new_x
import time
import utils
from utils import _get_cafe_polygon, set_size_plots
from scipy.stats import pearsonr
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


@torch.no_grad()
def load_model_and_get_dataloaders(experiment_config: dict, return_test_loader=False,
                                   get_only_model=False):
    timestamp = experiment_config['experiment']['timestamp'] if 'timestamp' in experiment_config.keys() else 'unknown'
    print(f'evaluating model from experiment with timestamp {timestamp}')

    DATADIR = experiment_config['data_directory']

    pred_stepsize = experiment_config['pred_stepsize']
    model = experiment_config['model'](**experiment_config['model_params'],
                                       dynamic_channels=experiment_config['dynamic_channels'],
                                       data_dim=experiment_config['data_dim'],
                                       pred_stepsize=pred_stepsize, )
    try:
        load_result = model.load_state_dict(
            torch.load(
                os.path.join('models', 'state_dicts', 'final/' +
                                                      experiment_config['experiment']['state_dict_fname']
                             )
            )
        )
    except FileNotFoundError:
        print('could not find model in final folder, falling back to last')
        load_result = model.load_state_dict(torch.load(os.path.join('models', 'state_dicts',
                                                      'last/'+ experiment_config['experiment']['state_dict_fname'])))

    if get_only_model:  # for backward compatibility with old construction where we always got the data
        num_ignored = 2 + return_test_loader
        return model, pred_stepsize, *[None for _ in range(num_ignored)]

    dataloader, val_dataloader, test_loader = experiment_config['dataset'].get_data_loaders(experiment_config,
                                                                                                    distinguish_flowing=
                                                                                                    experiment_config[
                                                                                                        'distinguish_flowing'],
                                                                                                    use_initial_pos_as_feature=
                                                                                                    experiment_config[
                                                                                                        'use_initial_pos_as_feature'])

    if not return_test_loader:
        return model, pred_stepsize, dataloader, val_dataloader

    return model, pred_stepsize, dataloader, val_dataloader, test_loader


@torch.no_grad()
def model_rollout(model: nn.Module, data_batch: torch.Tensor, pred_stepsize: int, rollout_length: int,
                  start_time: int = 0, use_posterior_sampling=False, verbose=False,
                  use_ground_truth_input=False, return_loglik=False, return_dists=False, miscellaneous_all=tuple(),
                  run_with_error_handling=True):

    # starting x, possibly including nans for thsoe that are not observed at this time
    x_prelim = data_batch[range(data_batch.size(0)), :, start_time]

    # miscellaneous info about the data (batch_idx)
    for i, o in enumerate(miscellaneous_all):
        o = o.to(x_prelim.device)
        miscellaneous_all[i] = o

    miscellaneous = [torch.ones_like(m) * m for m in miscellaneous_all]

    trues = [x_prelim.cpu().numpy()]
    preds = [x_prelim.cpu().numpy()]
    preds_cont = [x_prelim.cpu().numpy()]
    dists = [torch.distributions.Normal(loc=x_prelim.cpu(), scale=1e-6,
                                        validate_args=False)] if return_dists else []  # allow for NaN here

    kl = 0
    reconstr_loglik = 0

    start = time.time()
    for step in range(1, rollout_length + 1):
        target_time = start_time + pred_stepsize * step
        if np.any(target_time >= data_batch.shape[2]) and not use_posterior_sampling:
            # if the target time is outside the data:
            y_prelim = None
        else:
            # get the target value at the target time, append to true trajectory
            y_prelim = data_batch[range(data_batch.size(0)), :, target_time]
            trues.append(y_prelim.cpu().numpy())
        with torch.no_grad():
            x, y, idx_to_keep, idx_to_pred, miscellaneous = filter_input(
                x_prelim, y_prelim, miscellaneous)
            y = y if use_posterior_sampling else None  # set y to None if we are not using posterior sampling (ie inference)
            try:
                # one forward step
                pred_dist, add_loss_dict_step, pred_sample, *miscellaneous = model(x, y, *miscellaneous)
            except Exception as e:
                if run_with_error_handling:
                    print(repr(e), f'aborting further rollout at step {step}!')
                    trues = trues[:-1]  # remove the last element in trues because we could not add a pred
                    break
                else:
                    raise
            if return_loglik:
                # calculate log-likelihood on the go
                assert y is not None, 'use posterior sampling should be set to true for loglik calculation!'
                kl += add_loss_dict_step['KL'].item()
                reconstr_loglik_step = pred_dist.log_prob(y)
                reconstr_loglik += reconstr_loglik_step.item()

            # make the new input
            pred_sample, _ = construct_new_x(y_prelim, pred_sample, idx_to_keep, idx_to_pred,
                                                 other_prelim_gt=miscellaneous_all,
                                                 other_sample=miscellaneous)
            pred_mean, miscellaneous = construct_new_x(y_prelim, pred_dist.mean, idx_to_keep, idx_to_pred,
                                                       other_prelim_gt=miscellaneous_all,
                                                       other_sample=miscellaneous)
            preds.append(pred_sample.cpu().numpy())
            preds_cont.append(pred_mean.cpu().numpy())
            if return_dists:
                dists.append(pred_dist)

        x_prelim = torch.ones_like(
            pred_sample) * pred_sample  # quick hack to avoid modifying random_x in pred list inplace
        if use_ground_truth_input:
            x_prelim = y_prelim  # resort to one-step predictions -- set the next input to the ground truth observed state

    stop = time.time()
    if verbose:
        print(f'model rollout took {stop - start} seconds')

    if return_loglik:
        return trues, preds, preds_cont, reconstr_loglik, kl
    elif return_dists:
        return trues, preds, preds_cont, dists
    else:
        return trues, preds, preds_cont


def get_nonflowing_nodes_assuming_symmetry(data_batch, pred_sample, pred_sample_before, t: int, t_next, batch_idx, config):

    # preliminary checks and masks:
    b, c, time_shape, f = data_batch.shape
    b1, c1, f1 = pred_sample.shape
    assert b1 == b, 'batch sizes should match'
    pedes_mask = (pred_sample[:, 0, 4] == 1)
    flowing_mask = (pred_sample[:, 0, 7] == 1)
    # we should have one flowing for each non-flowing pedes (1 neighbor). since flowing are also pedes, we have 2*flowing = pedes:
    assert (pedes_mask.sum() == flowing_mask.sum() * 2), 'each non-flowing pedes should have one flowing neighbor'
    non_nan_mask = ~torch.isnan(data_batch[:, 0, t, 0])

    # now, get the acceleration of the prediction from the position information:
    pred_sample_real_coor = utils.to_real_coordinates(
        pred_sample[..., :4], True, True
    )
    pred_sample_before_real_coor = utils.to_real_coordinates(
        torch.from_numpy(pred_sample_before[..., :4]), True, True
    ).to(pred_sample_real_coor.device)
    delta_t = 1. / 10. * config['pred_stepsize']
    all_acc = (
                      (pred_sample_real_coor[..., 0:2] - pred_sample_before_real_coor[..., 0:2]) / 10. / delta_t - pred_sample_before_real_coor[...,
                                                                                              2:4]
              ) / delta_t

    # now, assume point symmetry for the non-flowing pedestrians:
    # if we are shy of other work to do (haha) it might be nice to vectorize the below code.
    # should be doable but we need to carefully check if everything aligns along the batch dimension.
    for b_id in batch_idx.unique():
        batch_mask = b_id == batch_idx
        data_batch_this = data_batch[batch_mask]
        ped_mask_this = pedes_mask[batch_mask]
        flow_mask_this = flowing_mask[batch_mask]
        nonflow_mask_this = torch.bitwise_and(~flow_mask_this, ped_mask_this)
        assert flow_mask_this.sum() == ped_mask_this.sum() / 2 == 1, 'expected one non-flowing ped and one flowing ped'
        non_nan_mask_this = non_nan_mask[batch_mask]
        acc_this = all_acc[batch_mask]
        acc_point_symmetric = acc_this[flow_mask_this][:,0] * -1
        # update the data_batch with euler step:
        data_batch_this[..., :4] = utils.to_real_coordinates(data_batch_this[..., :4], True, True)
        data_batch_this[nonflow_mask_this, 0, t+1, 2:4] = data_batch_this[nonflow_mask_this, 0, t, 2:4] + acc_point_symmetric * delta_t
        data_batch_this[nonflow_mask_this, 0, t+1, 0:2] = data_batch_this[nonflow_mask_this, 0, t, 0:2] + data_batch_this[nonflow_mask_this, 0, t+1, 2:4] * delta_t * 10
        data_batch_this[..., :4] = utils.to_model_internal_coordinates(data_batch_this[..., :4], True, True)
        data_batch_this[~nonflow_mask_this, :, t+1, :] = pred_sample[batch_mask][~nonflow_mask_this]
        data_batch[batch_mask] = data_batch_this

    return data_batch





@torch.no_grad()
def model_rollout_assume_symmetry(model: nn.Module, data_batch: torch.Tensor, pred_stepsize: int, rollout_length: int,
                  start_time: int = 0, use_posterior_sampling=False, verbose=False,
                  use_ground_truth_input=False, return_loglik=False, return_dists=False, miscellaneous_all=tuple(),
                  run_with_error_handling=True, config=None):

    x_prelim = data_batch[range(data_batch.size(0)), :, start_time]

    for i, o in enumerate(miscellaneous_all):
        o = o.to(x_prelim.device)
        miscellaneous_all[i] = o

    miscellaneous = [torch.ones_like(m) * m for m in miscellaneous_all]

    trues = [x_prelim.cpu().numpy()]
    preds = [x_prelim.cpu().numpy()]
    preds_cont = [x_prelim.cpu().numpy()]
    dists = [torch.distributions.Normal(loc=x_prelim.cpu(), scale=1e-6,
                                        validate_args=False)] if return_dists else []  # allow for NaN here

    kl = 0
    reconstr_loglik = 0

    start = time.time()
    for step in range(1, rollout_length + 1):
        target_time = start_time + pred_stepsize * step
        if np.any(target_time >= data_batch.shape[2]) and not use_posterior_sampling:
            y_prelim = None
        else:
            y_prelim = data_batch[range(data_batch.size(0)), :, target_time]
        with torch.no_grad():
            x, y, idx_to_keep, idx_to_pred, miscellaneous = filter_input(
                x_prelim, y_prelim, miscellaneous)
            y = y if use_posterior_sampling else None
            try:
                pred_dist, add_loss_dict_step, pred_sample, *miscellaneous = model(x, y, *miscellaneous)
            except Exception as e:
                if run_with_error_handling:
                    print(repr(e), f'aborting further rollout at step {step}!')
                    trues = trues[:-1]  # remove the last element in trues because we could not add a pred
                    break
                else:
                    raise
            if return_loglik:
                assert y is not None, 'use posterior sampling should be set to true for loglik calculation!'
                kl += add_loss_dict_step['KL'].item()
                reconstr_loglik_step = pred_dist.log_prob(y)
                reconstr_loglik += reconstr_loglik_step.item()

            pred_sample, trash = construct_new_x(y_prelim, pred_sample, idx_to_keep, idx_to_pred,
                                                 other_prelim_gt=miscellaneous_all,
                                                 other_sample=miscellaneous)
            pred_mean, miscellaneous = construct_new_x(y_prelim, pred_dist.mean, idx_to_keep, idx_to_pred,
                                                       other_prelim_gt=miscellaneous_all,
                                                       other_sample=miscellaneous)

            # update the data batch assuming point symmetry:
            data_batch = get_nonflowing_nodes_assuming_symmetry(data_batch, pred_sample, preds[-1],
                                                                target_time - pred_stepsize, target_time,
                                                                miscellaneous[0], config
                                                                )

            pred_sample = data_batch[range(data_batch.size(0)), :, target_time]
            preds.append(pred_sample.cpu().numpy())
            preds_cont.append(pred_mean.cpu().numpy())
            trues.append(data_batch[range(data_batch.size(0)), :, target_time].cpu().numpy())
            if return_dists:
                dists.append(pred_dist)

        x_prelim = torch.ones_like(
            pred_sample) * pred_sample  # avoid modifying random_x in pred list inplace
        if use_ground_truth_input:
            x_prelim = y_prelim

    stop = time.time()
    if verbose:
        print(f'model rollout took {stop - start} seconds')

    if return_loglik:
        return trues, preds, preds_cont, reconstr_loglik, kl
    elif return_dists:
        return trues, preds, preds_cont, dists
    else:
        return trues, preds, preds_cont





def moving_average(x, w, axis=-1):
    return np.apply_along_axis(np.convolve, axis=axis, arr=x, v=np.ones(w) / w, mode='same')

def calculate_auto_corr_single_lag(trajectories, k=1, feat_idx=0,
                                   min_avg_density=-np.inf, max_avg_density=np.inf, all_densities=None):
    # we see as all samples the pairs of observations spaced k lags apart that we have in the dataset
    # shape of trajectories:
    left = []
    right = []
    for i, t_temp in enumerate(trajectories):
        t = np.copy(t_temp)
        t[t[..., 7] != 1] = np.nan  # only keep flowing
        t = t[..., feat_idx]
        # t: (p, 1, t)
        t_lagged = np.roll(t, shift=k, axis=2)[..., k:]
        t_trunc = t[..., k:]
        if all_densities is not None:  # filter by density
            densities = all_densities[i]
            dens_rolling_avg = moving_average(densities, w=k, axis=-1) if k > 0 else densities
            dens_avg_aligned = np.roll(dens_rolling_avg, shift=k//2, axis=1)[..., k:][:, None]
            mask = np.bitwise_and(dens_avg_aligned >= min_avg_density, dens_avg_aligned <= max_avg_density)
            t_lagged[~mask] = np.nan
            t_trunc[~mask] = np.nan
        left.append(t_lagged[:, 0])
        right.append(t_trunc[:, 0])

    left = np.reshape(np.concatenate(left, axis=0), -1)
    right = np.reshape(np.concatenate(right, axis=0), -1)
    mask = np.bitwise_and(~np.isnan(left), ~np.isnan(right))
    if mask.sum() > 1:
        left = left[mask]
        right = right[mask]
        return pearsonr(left, right)[0]
    else:
        return np.nan


def calculate_auto_corr(trajectories, max_lag=50, feat_idx=0,
                        min_avg_density=-np.inf, max_avg_density=np.inf, all_densities=None):
    # we calculate the pearson corr between all samples that we have, for k lags apart
    corrs = []
    for k in range(max_lag):
        corrs.append(
            calculate_auto_corr_single_lag(trajectories, k, feat_idx=feat_idx,
                                           min_avg_density=min_avg_density, max_avg_density=max_avg_density,
                                           all_densities=all_densities
                                           )
        )
    return corrs



def calculate_local_density(data, how='voronoi'):
    """
    calculate the 'local' density of each node
    Parameters
    ----------
    data -- the data of shape (num_people, 1, time, 2) where the last dimension is (x,y). Expects only the pedestrian
    data, the geometry nodes and exit nodes should be filtered out already
    how -- how the local density is calculated.
    'voronoi' calculates it as the inverse of the area of the voronoi cell
    'radius' uses the radius of the circle around each node that is used to calculate the density (in m)
    radius -- the radius of the circle around each node that is used to calculate the density when how=='radius'

    Returns -- array of local densities with shape (people, t)
    -------

    """
    # convert the data to meters:
    data = (np.ones_like(data) * data)[:, 0, :, :]
    data = data / 1000

    # initialize output object:
    all_densities = np.zeros((data.shape[0], data.shape[1]))  # (people, t)

    # get cafe polygon which we need for filtering and density calculations:
    cafe_poly = _get_cafe_polygon()

    # iterate over the time dimension
    for t in range(data.shape[1]):
        data_t = np.ones_like(data[:, t, :]) * data[:, t, :]  # shape (num_people, 2)
        # filter out nan values at this time:
        nan_mask = np.isnan(data_t).any(axis=1)
        data_t = data_t[~nan_mask]

        if (~nan_mask).sum() == 0:
            all_densities[:, t][nan_mask] = np.nan
            continue  # skip to prevent voronoi error

        if how == 'voronoi':
            from scipy.spatial import Voronoi, voronoi_plot_2d
            from shapely.geometry import Polygon
            # take the geometry into account
            # reflect the pedestrian nodes along the boundaries of the domain s.t. personal space ends at the domain boundaries
            reflected_points = _get_artificial_nodes_for_voronoi(data_t)
            num_nodes_artificial = reflected_points.shape[0] # bookkeeping for later
            data_t = np.concatenate([data_t, reflected_points], axis=0)

            vor = Voronoi(data_t)
            areas = np.zeros((~nan_mask).sum())
            for point_idx, region_idx in enumerate(vor.point_region[:-num_nodes_artificial]):
                region = vor.regions[region_idx]
                if not -1 in region:  # bounded cell -- -1 indicates that the area is infinite
                    polygon = [vor.vertices[i] for i in region]
                    polygon_subtracted = Polygon(polygon).difference(cafe_poly)  # subtract the cafe from the personal space
                    areas[point_idx] = polygon_subtracted.area
                else:
                    areas[point_idx] = np.nan
            density = 1 / areas
            density = np.nan_to_num(density, posinf=np.nan)
            all_densities[:, t][~nan_mask] = density
            all_densities[:, t][nan_mask] = np.nan
        else:
            raise NotImplementedError(f'how={how} not implemented')
    return all_densities


def _get_geom_nodes_for_voronoi(num_nodes_track=100, num_nodes_cafe=48):
    # units in m
    ys_track = np.linspace(-2000, 75000, num=num_nodes_track)
    x_track1 = np.linspace(3900, 3000, num=num_nodes_track)
    x_track2 = np.linspace(-8500, -9400, num=num_nodes_track)
    track_nodes_1 = np.stack([x_track1, ys_track], axis=1)
    track_nodes_2 = np.stack([x_track2, ys_track], axis=1)
    track_nodes = np.concatenate([track_nodes_1, track_nodes_2], axis=0) / 1000

    # cafe consists of 4 sides, we put 10/12th of the nodes on the top and bottom sides which are longer
    ys_cafe = np.concatenate([
        np.linspace(28000, 50000, num=int(num_nodes_cafe * 5 / 12)),
        np.linspace(28000, 50000, num=int(num_nodes_cafe * 5 / 12)),
        np.linspace(28000, 28000, num=max(int(num_nodes_cafe * 1 / 12), 3))[1:-1],
        np.linspace(50000, 50000, num=max(int(num_nodes_cafe * 1 / 12), 3))[1:-1]
    ])
    x_cafe = np.concatenate([
        np.linspace(-500, -700, num=int(num_nodes_cafe * 5 / 12)),
        np.linspace(-5000, -5200, num=int(num_nodes_cafe * 5 / 12)),
        np.linspace(-500, -5000, num=max(int(num_nodes_cafe * 1 / 12), 3))[1:-1],
        np.linspace(-700, -5200, num=max(int(num_nodes_cafe * 1 / 12), 3))[1:-1]
    ])

    cafe_nodes = np.stack([x_cafe, ys_cafe], axis=1) / 1000

    return np.concatenate([track_nodes, cafe_nodes], axis=0)


def _get_artificial_nodes_for_voronoi(data):
    # reflect all nodes along the bounds of the domain
    # data is of shape (num_people, 2)
    # we want to reflect along y=-2 as well as y = 75
    # we also want to reflect along x=3.45 and x=-8.95
    y_reflects = [-2, 75]
    x_reflects = [3.45, -8.95]

    reflected_points = []
    for y in y_reflects:
        data_reflected = np.ones_like(data) * data
        data_reflected[:, 1] = (data_reflected[:, 1] - y) * -1 + y
        reflected_points.append(data_reflected)

    for x in x_reflects:
        data_reflected = np.ones_like(data) * data
        data_reflected[:, 0] = (data_reflected[:, 0] - x) * -1 + x
        reflected_points.append(data_reflected)
    reflected_points = np.concatenate(reflected_points, axis=0)
    return reflected_points


def to_coordinate_frame(old_coordinates, origin, north):
    """
    :param old_coordinates: np.array of shape (*batch_dims, 2, ) representing coordinates in original coordinate system
    :param origin: np.array of shape (*batch_dims, 2, ) representing coordinates of origin in original coordinate system
    :param north: np.array of shape (*batch_dims, 2, ) representing direction of north in original coordinate system
    :returns: np.array of shape (*batch_dims, 2, ) representing coordinates in new coordinate system where north is [0, 1]
    """
    north_normalized = north / np.linalg.norm(north, axis=-1, keepdims=True, ord=2)
    east_normalized = np.stack([north_normalized[..., 1], -north_normalized[..., 0]], axis=-1)
    rotation_matrix = np.stack([east_normalized, north_normalized], axis=-2)

    old_coordinates_from_origin = old_coordinates - origin
    new_coordinates = (rotation_matrix @ old_coordinates_from_origin[..., None]).squeeze(-1)
    return new_coordinates


def get_distance_matrix(from_coordinates, to_coordinates, mask_self_distances=True):
    """get_distance_matrix
    return a matrix of distances between from_coordinates to to_coordinates

    :param from_coordinates: np.array of shape (*batch_dims, index_from, 2) representing coordinates of starting points
    :param to_coordinates: np.array of shape (*batch_dims, index_to, 2) representing coordinates of ending points
    :param mask_self_distances: bool, whether to mask distances between points with the same index_from and index_to to nan
    :returns: np.array of shape (*batch_dims, index_from, index_to) representing distances between starting and ending points
    """
    from_coordinates = from_coordinates[..., None, :]
    to_coordinates = to_coordinates[..., None, :, :]
    distances = np.linalg.norm(from_coordinates - to_coordinates, axis=-1, ord=2)
    if mask_self_distances:
        mask = distances == 0  # note we cannot mask the diagonal since the distmat is not calculated with itself
        distances = np.where(mask, np.nan, distances)
    return distances


def get_nearest_neighbor_coordinates(to_coordinates, distance_matrix):
    """get_nearest_neighbor_coordinates
    return the coordinates of the nearest neighbor of each coordinate in to_coordinates

    :param to_coordinates: np.array of shape (*batch_dims, index_to, 2) representing coordinates of ending points
    :param distance_matrix: np.array of shape (*batch_dims, index_from, index_to) representing distances between starting and ending points
    :returns: np.array of shape (*batch_dims, index_to, 2) representing coordinates of nearest neighbors
    """
    distance_matrix_temp = np.nan_to_num(distance_matrix, nan=np.inf)
    not_all_nan = ~np.all(
        np.isnan(distance_matrix),
        axis=-1
        )  # (*batch_dims, index_from) -- at times where somebody is not present, the distmat is only nans. but this person has no neighbors
    nearest_neighbor_indices = np.argmin(distance_matrix_temp, axis=-1)
    nearest_neighbor_coordinates = np.take_along_axis(to_coordinates, nearest_neighbor_indices[..., None], axis=1)
    nearest_neighbor_coordinates[~not_all_nan] = np.nan  # argmin returns 1 for nan slices, so we need to mask those
    return nearest_neighbor_coordinates


def get_radial_distance_distribution(distance_matrix, bins=100, max_distance=2, mask=None):
    """get_radial_distance_distribution

    :param distance_matrix: np.array of shape (*batch_dims, index_from, index_to) representing distances between starting and ending points
    :param bins: int, number of bins to divide distances into
    :param max_distance: float, maximum distance to consider
    :param mask: optional np.array of shape (*batch_dims, index_from) representing whether to consider each starting point
    :returns: tuple of two arrays:
        np.array of shape (bins, ) representing mean number of points within each radial distance
        np.array of shape (bins, ) representing standard deviation of number of points within each radial distance
    """
    distance_matrix = distance_matrix[..., None]
    distance_bins = np.expand_dims(
        np.linspace(max_distance / bins, max_distance, bins),
        axis=tuple(range(len(distance_matrix.shape) - 1))
        )
    distance_less_than_bin = distance_matrix <= distance_bins  # (*batch_dims, index_from, index_to, bins)
    if mask is not None:
        mask = mask[..., None, None]  # (*batch_dims, index_from, 1, 1)
        distance_less_than_bin = np.where(mask, distance_less_than_bin, np.nan)
    num_points_within_distance = np.nansum(distance_less_than_bin, axis=-2)  # (*batch_dims, index_from, bins)
    reducible_dims = tuple(range(len(num_points_within_distance.shape) - 1))
    mean = np.nanmean(num_points_within_distance, axis=reducible_dims)
    std = np.nanstd(num_points_within_distance, axis=reducible_dims)
    return mean, std


def add_alpha_to_cmap(cmap, N=100):
    base_colors = cmap(np.linspace(0., 1., N))
    base_colors[:, -1] = np.linspace(0., 1., N)
    return ListedColormap(base_colors)


def _kd_func(x, points, bandwidth=.05):
        """
        kernel density function
        :param x: (*batch_size, 2) array of coordinates to evaluate at
        :param points: 
        :returns: (*batch_size, ) array of densities between 0. and 1.
        """
        batch_dims = x.shape[:-1]
        point_dims = points.shape[:-1]
        x = np.expand_dims(x, axis=tuple(range(len(point_dims))))
        points_ = np.expand_dims(points, axis=tuple(range(len(point_dims), len(point_dims) + len(batch_dims))))
        dif_square = np.sum(np.square(x - points_), axis=-1)
        raw_values = np.sum(np.exp(-dif_square / (2 * bandwidth ** 2)), axis=tuple(range(len(point_dims))))
        return raw_values / np.max(raw_values)

def density_plot_2d(points, max_radius, grid_size=(100, 100), cmap=None, eval_func=_kd_func):
    # polar plot of nearest neighbor
    if cmap is None:
        cmap = add_alpha_to_cmap(colormaps['Reds'])

    grid = np.stack(np.meshgrid(
        np.linspace(-max_radius, max_radius, grid_size[0]),
        np.linspace(-max_radius, max_radius, grid_size[1]),
    ), axis=-1)
    values_at_grid = eval_func(grid, points)

    # fig = plt.figure(figsize=(2.5, 2.5))
    fig = plt.figure(figsize=set_size_plots(397, fraction=1 / 2.25 * 9 / 10 * 3/5, h_to_w_ratio=1))


    ax_img = fig.add_axes((0, 0, 1, 1), label='img', frame_on=False, xticks=[], yticks=[])
    ax_img.imshow(values_at_grid, cmap=cmap, origin='lower', interpolation='bilinear')

    ax_polar = fig.add_axes((0, 0, 1, 1), projection='polar', label='polar')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rticks([max_radius/3, 2/3*max_radius, max_radius])
    ax_polar.set_rlim(0, max_radius)
    ax_polar.patch.set_alpha(0)
    for gl in ax_polar.yaxis.get_gridlines():
        gl.set_color('black')
        gl.set_alpha(0.25)
    for gl in ax_polar.xaxis.get_gridlines():
        gl.set_color('black')
        gl.set_alpha(0.25)

    SHOULDER_WIDTH = .45
    SHOULDER_DEPTH = .17
    HEAD_LENGTH = .2
    HEAD_WIDTH = .17
    NOSE_WIDTH = .02
    NOSE_LENGTH = .03

    left_shoulder_centre_x = -SHOULDER_WIDTH/2 + SHOULDER_DEPTH/2

    person = [
        patches.Rectangle((left_shoulder_centre_x, -SHOULDER_DEPTH/2), SHOULDER_WIDTH-SHOULDER_DEPTH, SHOULDER_DEPTH, facecolor='#94dfff'),  # Main body
        patches.Circle((left_shoulder_centre_x, -0.), SHOULDER_DEPTH/2, facecolor='#94dfff'),  # left shoulder
        patches.Circle((-left_shoulder_centre_x, -0.), SHOULDER_DEPTH/2, facecolor='#94dfff'),  # right shoulder
        patches.Ellipse((0., 0.), HEAD_WIDTH, HEAD_LENGTH, facecolor='#7ea6e0'),
        patches.Ellipse((0., HEAD_LENGTH/2), NOSE_WIDTH, NOSE_LENGTH, facecolor='#7ea6e0')
    ]
    # add a 'person' to the centre of the plot:
    ax_person = fig.add_axes((0, 0, 1, 1), label='person', frame_on=False, xticks=[], yticks=[])
    ax_person.set_xlim(-max_radius, max_radius)
    ax_person.set_ylim(-max_radius, max_radius)
    ax_person.set_facecolor('none')
    ax_person.add_collection(collections.PatchCollection(person, match_original=True))

    return fig


def plot_nn_heat_map(
        nn_coordinates_in_local_frame, 
        max_radius=2.,
        grid_size=(50, 50), 
        title=None, 
        show_plot=True, 
        xlabel=None,
        eval_func=_kd_func,
        cmap=None
        ):
    nn_coordinates_in_local_frame = nn_coordinates_in_local_frame[~np.isnan(nn_coordinates_in_local_frame).any(axis=1)]
    fig = density_plot_2d(
        nn_coordinates_in_local_frame, 
        max_radius=max_radius, 
        grid_size=grid_size, 
        eval_func=eval_func,
        cmap=cmap
        )
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if show_plot:
        plt.show()
    return fig


def _get_crowd(num_crowd, T, cfg, as_rows=False, lattice_x_max=3000, xrange=(-500, 3500), yrange=(28000, 50000),
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    # X crowd in range (-500, 3500) in real coordinates
    # Y crowd in range (28000, 50000) in real coordinates
    if not as_rows:
        X_crowd = (torch.rand(num_crowd) * (xrange[1] - xrange[0]) + xrange[0])[:, None, None]
        Y_crowd = (torch.rand(num_crowd) * (yrange[1] - yrange[0]) + yrange[0])[:, None, None]
    else:
        # put the X_crowd at 0 and 3000
        Y_crowd = torch.cat([torch.linspace(yrange[0], yrange[1], num_crowd // 2) for _ in range(2)])
        X_crowd = torch.cat([torch.zeros(num_crowd//2), torch.ones(num_crowd//2) * lattice_x_max])
        X_crowd = X_crowd[:, None, None]
        Y_crowd = Y_crowd[:, None, None]

    crowd = torch.zeros(size=(X_crowd.shape[0], 1, T, 10))
    crowd[..., 0] = X_crowd
    crowd[..., 1] = Y_crowd
    crowd[..., 2] = 0
    crowd[..., 3] = 0
    crowd[..., 4] = 1
    crowd[..., 5:7] = 0
    crowd[..., 7] = 0

    crowd[..., :4] = utils.to_model_internal_coordinates(crowd[..., :4], cfg['standard_normalization'], True)

    crowd[..., 8:] = crowd[..., :2] if cfg['use_initial_pos_as_feature'] else 0.  # init pos as feat for pedes

    return crowd.to(device)


def get_nn_coordinates_in_local_frame(from_coordinates, from_velocities, to_coordinates, max_radius=None, ):
    distance_matrix = get_distance_matrix(from_coordinates, to_coordinates)

    nn_coordinates = get_nearest_neighbor_coordinates(to_coordinates, distance_matrix)  # (*batch_dims, from_indices, 2)
    nn_coordinates_in_local_frame = to_coordinate_frame(old_coordinates=nn_coordinates, origin=from_coordinates,
                                                        north=from_velocities)
    nn_coordinates_in_local_frame = nn_coordinates_in_local_frame.reshape(-1, 2)  # flatten all  but the last dimension
    nn_coordinates_in_local_frame = nn_coordinates_in_local_frame[~np.isnan(nn_coordinates_in_local_frame).any(axis=1)]

    if max_radius is not None:
        radius = np.linalg.norm(nn_coordinates_in_local_frame, axis=-1, ord=2)
        nn_coordinates_in_local_frame = nn_coordinates_in_local_frame[radius <= max_radius]

    return nn_coordinates_in_local_frame


# functions to check if two line segments overlap
def orientation(triangles):
    """For each triangle, gives its orientation:
    0: points are collinear
    1: points are clockwise
    2: points are counterclockwise

    Parameters
    ----------
    triangles : ndarray of dimension >=2, with shape (..., 3, 2)
        2D Cartesian coordinates of points making up triangles.
        The second to last dimension indexes the three points of the triangle,
        last dimension indexes the x and y coordinate,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of integers (of same shape as input with the last two dimensions removed)
        orientation of each triangle, using broadcasting.
    """
    # gives order of points of a triangle:
    # 0 : collinear
    # 1 : clockwise
    # 2 : counterclockwise

    triangles = np.asarray(triangles)

    p = triangles[..., 0, :]  # 1st point of each triangle
    q = triangles[..., 1, :]  # 2nd point of each triangle
    r = triangles[..., 2, :]  # 3rd point of each triangle
    val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])

    val[val > 0] = 1
    val[val < 0] = 2

    return val.astype(int)


def between(vals):
    """"Check if value C is between values A and B.

    Parameters
    ----------
    vals : ndarray of dimension >=1, with shape (..., 3)
        The second to last dimension indexes the values A, B, C,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last dimension removed)
        Whether value C is between values A and B (not equal to either), regardless of whether A is larger than B or vice versa.
    """
    bools1 = vals[..., 2] < np.maximum(vals[..., 0], vals[..., 1])
    bools2 = vals[..., 2] > np.minimum(vals[..., 0], vals[..., 1])
    return bools1 * bools2


def onSegment(points):
    """Check if a point C is on a line segment from point A to B.

    Parameters
    ----------
    vals : ndarray of dimension >=2, with shape (..., 3, 2)
        2D Cartesian coordinates of points defining pairs of line segments.
        The second to last dimension indexes the points A, B, C,
        last dimension indexes the x and y coordinate,
        broadcasted over remaining dimensions.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last two dimensions removed)
        Whether C is on line segment AB, but not the same as either A or B.
    """
    points = np.asarray(points)

    boolsx = between(points[..., 0])
    boolsy = between(points[..., 1])

    return boolsx + boolsy


def intersect(points):
    """Check whether a line segment from point A to point B
    intersects a line segment from point C to point D (in 2D Cartesian coordinates). (Only the end points coinciding does not count.)

    Parameters
    ----------
    points : ndarray of dimension >=2, with shape (..., 4, 2)
        2D Cartesian coordinates of points defining pairs of line segments.
        The second to last dimension indexes the four points A, B, C, D,
        last dimension indexes the x and y coordinate.

    Returns
    -------
    ndarray of booleans (of same shape as input with the last two dimensions removed)
        Whether the line segments intersect or not.
    """
    points = np.asarray(points)  # shape [..., 4, 2]

    # different combinations of points to take as triangles
    combis = [[0, 1, 2], [0, 1, 3], [2, 3, 0], [2, 3, 1]]

    # orientation of each triangle
    ori = orientation(points[..., combis, :])

    # line segments cross and nothing is collinear
    bools = (ori[..., 0] != ori[..., 1]) * (ori[..., 2] != ori[..., 3]) * (ori != 0).all(axis=-1)

    # collinear combinations
    bools_coll = ori == 0

    # any one of the combinations is collinear
    bools_coll2 = bools_coll.any(axis=-1)

    # all cases where 3 or more points are collinear
    coll_points = points[bools_coll2]

    # check overlap in case of collinear points
    bools_onSeg = onSegment(coll_points[..., combis, :])

    # if collinear and onSegment, then also return True
    bools_coll[bools_coll2] *= bools_onSeg

    return bools + bools_coll.any(axis=-1)


def line_polygon_overlap(line_points, polygon_points):
    """Indicates whether lines and a polygon overlap

    Parameters
    ----------
    line_points : ndarray-like, shape (..., 2, 2)
        line_points[..., 0, :]: x and y coordinate of start point of each line
        line_points[..., 1, :]: x and y coordinate of end point of each line
    polygon_points : ndarray-like, shape (n, 2)
        list of 2D points defining a polygon with n corners (must be clockwise or counterclockwise)

    Returns
    ----------
    bools : ndarray
        for each line, whether it overlaps with the polygon. Will have the same shape as line_points, minus the last two dimensions
    """
    line_points = np.asarray(line_points)
    polygon_points = np.asarray(polygon_points)

    n_pp = len(polygon_points)
    shift = (np.arange(n_pp) + 1) % n_pp
    segments2 = np.stack((polygon_points, polygon_points[shift]), axis=1)
    br_arrs = np.broadcast_arrays(line_points[..., np.newaxis, :, :], segments2)
    temp = np.concatenate(br_arrs, axis=-2)
    bools = intersect(temp)

    return bools.any(axis=-1)


def _calc_acc_on_pedestrians(config, model: nn.Module, data: torch.Tensor, *miscellaneous_model_args):
    # input data must be in model internal coors
    pred_dist, add_loss_dict_step, pred_sample, *miscellaneous = model(data, None, *miscellaneous_model_args)
    pred_sample_real_coor = utils.to_real_coordinates(pred_sample[..., :4], True, True)
    data_real_coor = utils.to_real_coordinates(data[..., :4], True, True)
    delta_t = 1. / 10. * config['pred_stepsize']

    # reverse the Euler step to get the acceleration based on positions and input vel
    # this also takes edge correction and pos/vel correction into account
    # note: coors in mm, vel in cm/s
    # so we divide coors by 10 to go to cm as well
    net_acc = (  # new v (derived from position difference, /10 to go to cm/s) minus old v (in cm/s), divided by delta_t to get acc
                      (pred_sample_real_coor[..., 0:2] - data_real_coor[..., 0:2]) / 10. / delta_t - data_real_coor[..., 2:4]
              ) / delta_t
    return net_acc  # (nodes, 1, [acc_x, acc_y]) in cm/s^2

def get_synth_setup_crowd_for_all_loc(cfg, data_auxiliary, crowd_range=(1, 20), crowd_com_dist_x=0,
                                   crowd_com_dist_y=-2000,
                                   crowd_std=500,
                                   x_vel=0,
                                   y_vel=-100,
                                   x_vel_crowd=0, y_vel_crowd=0,
                                   num_x=10, num_y=10, xrange=None, yrange=None, get_crowd_as_lattice=False,
                                      crowd_lattice_width=2000, add_spacing_center_lattice=0.):
    # output is returned in model internal coordinates!
    if xrange is None:
        xrange = (-8500, 3000)
    if yrange is None:
        yrange = (10000, 75000)

    # generate pedes coordinates:
    XX, YY = np.meshgrid(np.linspace(*xrange, num_x), np.linspace(*yrange, num_y))
    XX = XX.reshape(-1)
    YY = YY.reshape(-1)

    # generate the batches of synthetic test cases:
    all_test_data = []
    all_batch_idx = []
    all_crowd_sizes = []
    all_dx = []
    all_dy = []
    x_used = []
    y_used = []
    mask_used = [False for _ in range(XX.shape[0])]
    for i in range(XX.shape[0]):

        x = XX[i]
        y = YY[i]
        crowd_com_dist_x_i = crowd_com_dist_x[i] if isinstance(crowd_com_dist_x, np.ndarray) else crowd_com_dist_x
        crowd_com_dist_y_i = crowd_com_dist_y[i] if isinstance(crowd_com_dist_y, np.ndarray) else crowd_com_dist_y
        x_vel_crowd_i = x_vel_crowd[i] if isinstance(x_vel_crowd, np.ndarray) else x_vel_crowd
        y_vel_crowd_i = y_vel_crowd[i] if isinstance(y_vel_crowd, np.ndarray) else y_vel_crowd

        # generate artificial pedestrian nodes:
        crowd_size = int(np.random.randint(*crowd_range))
        pedes = torch.cat([torch.zeros_like(data_auxiliary[0:1]) for _ in range(crowd_size + 1)])
        pedes[..., 4] = 1  # pedes bool
        pedes[0, 7] = 1  # flowing bool -- only for first pedes (the one on which we do measurements as a function of the others)!

        # set coordinates and velocity for the first pedes:
        pedes[0, 0] = x
        pedes[0, 1] = y
        pedes[0, 2] = x_vel[i] if isinstance(x_vel, np.ndarray) else x_vel
        pedes[0, 3] = y_vel[i] if isinstance(y_vel, np.ndarray) else y_vel

        # set the coordinates of the neighboring pedes
        if not get_crowd_as_lattice:
            neighbor_coors = torch.randn(crowd_size, 2) * crowd_std
        else:  # we get the crowd as a lattice rather than random
            # for the lattice, we put people crowd_std mm apart. The width of the lattice (range in x_coor) is crowd_lattice_width mm
            # the depth is whatever is necessary to accommodate the crowd_size and respect the crowd_std distance
            people_per_col = crowd_lattice_width // crowd_std
            num_cols = int(np.ceil((crowd_range[-1] -1) / people_per_col))  # number of columns
            num_rows = min(people_per_col, crowd_size)
            neighbor_coors = torch.zeros(crowd_size, 2)
            xx_crowd, yy_crowd = torch.meshgrid(torch.linspace(-crowd_std*(num_rows-1) / 2, crowd_std*(num_rows-1) / 2, num_rows),
                                    torch.linspace(-crowd_std*(num_cols-1), 0, num_cols))
            # remove some coordinates from the lattice at random to get the right crowd size at random lattice locations:
            num_coors_to_remove = num_rows * num_cols - crowd_size
            # need to call contiguous since otherwise the tensor is a view of the same underlying tensor multiple times
            xx_crowd = xx_crowd.contiguous().reshape(-1)
            yy_crowd = yy_crowd.contiguous().reshape(-1)
            if num_coors_to_remove > 0:
                idx_to_keep = np.random.choice(xx_crowd.shape[0], crowd_size, replace=False)
                xx_crowd = xx_crowd[idx_to_keep]
                yy_crowd = yy_crowd[idx_to_keep]

            neighbor_coors[:, 0] = xx_crowd
            neighbor_coors[:, 1] = yy_crowd
            assert not torch.isnan(neighbor_coors).any()

        # now put the crowd at the right relative location of the pedestrian:
        neighbor_coors[:, 1] += crowd_com_dist_y_i + y
        neighbor_coors[:, 0] += crowd_com_dist_x_i + x
        spacing_center_lattice_this = add_spacing_center_lattice[i] if isinstance(add_spacing_center_lattice, np.ndarray) else add_spacing_center_lattice
        # make the extra spacing in the center of the lattice by moving every neighbor who has an x coordinate smaller than the person to the left, the others to the right:
        if spacing_center_lattice_this > 0:
            mask = neighbor_coors[:, 0] < x  # distinguish whether to move people up or down
            neighbor_coors[:,0][mask] -= spacing_center_lattice_this / 2.
            neighbor_coors[:,0][~mask] += spacing_center_lattice_this / 2.

        pedes[1:, 0:2] = neighbor_coors
        pedes[1:, 2] = x_vel_crowd_i
        pedes[1:, 3] = y_vel_crowd_i

        in_cafe = utils._check_if_inside_cafe(pedes, slack=0, in_mm=True)[0].long()
        if in_cafe.float().mean() > 0.05:  # more than 5 percent are in the cafe
            continue  # skip this setup

        # now preprocess the data for model processing:
        pedes[..., :4] = utils.to_model_internal_coordinates(pedes[..., :4], True,True)

        pedes[0, -2] = pedes[0, 0]
        pedes[0, -1] = pedes[0, 1] + 0.2  # 0.2*10m starting pos to the right of current pos
        pedes[1:, -2] = pedes[1:, 0]
        pedes[1:, -1] = pedes[1:, 1]

        all_data = torch.cat([pedes, data_auxiliary], dim=0).unsqueeze(1)
        batch_idx = torch.zeros_like(all_data[:, 0, 0]) + i

        # store output data
        all_test_data.append(all_data)
        all_batch_idx.append(batch_idx)
        all_crowd_sizes.append(torch.Tensor([crowd_size]).long())
        x_used.append(torch.Tensor([x]))
        y_used.append(torch.Tensor([y]))
        all_dx.append(torch.Tensor([crowd_com_dist_x_i]))
        all_dy.append(torch.Tensor([crowd_com_dist_y_i]))
        mask_used[i] = True

    # relevant output data to use:
    all_test_data = torch.cat(all_test_data, dim=0)
    all_batch_idx = torch.cat(all_batch_idx, dim=0)

    # some data that is used for bookkeeping:
    all_crowd_sizes = torch.cat(all_crowd_sizes, dim=0)
    all_dx = torch.cat(all_dx, dim=0)
    all_dy = torch.cat(all_dy, dim=0)
    x_used = torch.cat(x_used, dim=0)
    y_used = torch.cat(y_used, dim=0)
    mask_used = torch.Tensor(mask_used).bool()

    return all_test_data, all_batch_idx, all_crowd_sizes, all_dx, all_dy, x_used, y_used, mask_used


def force_scaling_once_for_all_loc(cfg, model, data_auxiliary, crowd_range=(1, 20), crowd_com_dist_x=0,
                                   crowd_com_dist_y=-2000,
                                   crowd_std=500,
                                   x_vel=0,
                                   y_vel=-100,
                                   x_vel_crowd=0, y_vel_crowd=0,
                                   num_x=10, num_y=10, get_crowd_as_lattice=False, crowd_lattice_width=2000,
                                   calculate_social_forcing=False, top_ks_social_forcing=tuple(), xrange=None,
                                   yrange=None, U0=0.375, R=2., max_angle_sf=np.pi,
                                   add_spacing_center_lattice=0.):
    """
    calculate the acceleration experienced by a pedestrian at each location, when a crowd of random size is located
    crowd_com_dist mm next to the pedestrian (negative indicates to the left), and the pedestrian is moving with x_vel and y_vel,
    and crowd people locations are from isotropic gaussian around the COM with std crowd_std. all units in mm, except for velocities, which are in cm/s
    :param cfg: config dict
    :param model: model
    :param data_auxiliary: auxiliary data to use for the model (virtual exit nodes)
    :param crowd_range: tuple of ints, range of crowd sizes to consider
    :param crowd_com_dist_x: x distance of the crowd from the pedestrian
    :param crowd_com_dist_y: y distance of the crowd from the pedestrian
    :param crowd_std: std of the crowd distribution (gaussian crowd) or spacing in the lattice (crowd on lattice locations)
    :param x_vel: x velocity of the pedestrian
    :param y_vel: y velocity of the pedestrian
    :param x_vel_crowd: x velocity of the crowd
    :param y_vel_crowd: y velocity of the crowd
    :param num_x: number of x locations to consider
    :param num_y: number of y locations to consider
    :param get_crowd_as_lattice: bool, whether to generate the crowd as a lattice
    :param crowd_lattice_width: width of the lattice in mm
    :param calculate_social_forcing: bool, whether to calculate social forces
    :param top_ks_social_forcing: tuple of ints, top k social forces to consider
    :param xrange: range of x coordinates to consider
    :param yrange: range of y coordinates to consider
    :param U0, R, max_angle_sf: social force parameters
    :param add_spacing_center_lattice: float, additional spacing in the center of the lattice crowd
    """

    all_test_data, all_batch_idx, all_crowd_sizes, all_dx, all_dy, x_used, y_used, mask_used = get_synth_setup_crowd_for_all_loc(
        cfg, data_auxiliary, crowd_range, crowd_com_dist_x, crowd_com_dist_y, crowd_std, x_vel, y_vel,
        x_vel_crowd, y_vel_crowd, num_x, num_y, xrange=xrange, yrange=yrange,
        get_crowd_as_lattice=get_crowd_as_lattice,
        crowd_lattice_width=crowd_lattice_width, add_spacing_center_lattice=add_spacing_center_lattice)


    all_acc = _calc_acc_on_pedestrians(cfg, model, all_test_data, all_batch_idx)
    all_batch_idx = all_batch_idx.cpu()
    all_acc = all_acc.cpu()
    # let's get the accelerations on the first element of each batch, which is the pedestrian we measure
    all_acc_on_pedes = []
    for b in all_batch_idx.unique():
        acc_b = all_acc[all_batch_idx == b]  # (n, 1, 2)
        acc = acc_b[0, 0]  #(2,)
        all_acc_on_pedes.append(acc.unsqueeze(0))

    all_acc_on_pedes = torch.cat(all_acc_on_pedes, dim=0)  # (n, 2)

    if not calculate_social_forcing:
        return (x_used, y_used), all_acc_on_pedes, all_crowd_sizes, all_dx, all_dy, mask_used

    # calculate social forces if desired:
    social_forces = []
    top_k_social_forces = {}
    for b in all_batch_idx.unique():
        test_data_this = all_test_data.cpu()[all_batch_idx == b]
        # first pedes is the pedestrian we calc the acc on:
        pedes = test_data_this[0:1]
        neighbors = test_data_this[1:][test_data_this[1:, 0, 4] == 1]  # only keep neighbors, and ditch virtual nodes
        social_force_this, top_k_social_force_this = social_forcing_acc(pedes[:, 0, :2] * 10,  # go to m
                                                                        neighbors[:, 0, :2] * 10,  # go to m
                                                                        top_ks=top_ks_social_forcing,
                                                                        U0=U0, R=R,
                                                                        max_angle=max_angle_sf)  #(1, neighbors, 2)
        social_forces.append(social_force_this.sum(dim=1) * 100)  # *100 -> cm/s^2
        if len(top_ks_social_forcing) > 0:
            for k, v in top_k_social_force_this.items():
                if k not in top_k_social_forces:
                    top_k_social_forces[k] = []
                top_k_social_forces[k].append(v.sum(dim=1) * 100)

    social_forces = torch.cat(social_forces, dim=0)
    if len(top_ks_social_forcing) > 0:
        for k, v in top_k_social_forces.items():
            top_k_social_forces[k] = torch.cat(v, dim=0)
            assert top_k_social_forces[k].shape[0] == social_forces.shape[0], 'expected same number of measurements for top-k as linear'

        return (x_used, y_used), all_acc_on_pedes, all_crowd_sizes, all_dx, all_dy, social_forces, top_k_social_forces, mask_used

    return (x_used, y_used), all_acc_on_pedes, all_crowd_sizes, all_dx, all_dy, social_forces, None, mask_used



def calc_acc_for_locs(cfg, model, XX, YY, data_auxiliary, initial_x_vel=None, initial_y_vel=None,
                      initial_vel_tensor=None):
    # calculate the acceleration on a pedestrian in the absence of any neighbors
    # note: all additional nodes (geometry, exit, etc) are already in data_auxiliary
    # also additional pedestrian neighbors can be put in data auxiliary, but then these NEED TO BE THE SAME
    # for all locations, so that the batch processing works (e.g. modifying the preferred velocity plot, but not
    # for the crowd force scaling plot
    # everything in real coors as input here
    # X, Y, and vel in real coordinates!

    assert (initial_x_vel is None) != (
                initial_vel_tensor is None), 'either initial_x_vel or inital_vel_tensor must be provided'
    assert (initial_y_vel is None) == (
                initial_x_vel is None), 'if initial_y_vel is provided, initial_x_vel must be provided as well'

    if initial_vel_tensor is None:
        initial_vel_tensor = torch.Tensor([[initial_x_vel, initial_y_vel]]).to(data_auxiliary.device).repeat(
            XX.shape[0], 1)  # (n_loc, 2)

    # generate the batches of synthetic test cases:
    all_test_data = []
    all_batch_idx = []
    for i in range(XX.shape[0]):

        x = XX[i]
        y = YY[i]
        init_vel = initial_vel_tensor[i]

        # generate artificial pedestrian node:
        pedes = torch.zeros_like(data_auxiliary[0:1])
        pedes[..., 4] = 1  # pedes bool
        pedes[..., 7] = 1  # flowing bool

        # set reasonable coordinates and velocity for the first pedes:
        pedes[0, ..., 0] = x
        pedes[0, ..., 1] = y
        pedes[0, ..., 2:4] = init_vel

        # now preprocess the data for model processing:
        pedes[..., :4] = utils.to_model_internal_coordinates(pedes[..., :4], True, True)

        # place of birth the same as current position
        pedes[0, ..., -2] = pedes[0, ..., 0]
        pedes[0, ..., -1] = pedes[0, ..., 1]

        all_data = torch.cat([pedes, data_auxiliary], dim=0)
        all_data = all_data.reshape(all_data.shape[0], 1, all_data.shape[-1])
        batch_idx = torch.zeros_like(all_data[:, 0, 0]) + i

        all_test_data.append(all_data)
        all_batch_idx.append(batch_idx)

    all_test_data = torch.cat(all_test_data, dim=0)
    all_batch_idx = torch.cat(all_batch_idx, dim=0)

    all_acc = _calc_acc_on_pedestrians(cfg, model, all_test_data, all_batch_idx)
    all_batch_idx = all_batch_idx.cpu()
    all_acc = all_acc.cpu()
    all_acc_on_pedes = []  # let's get the accelerations on the first element of each batch, which is the pedestrian we care about
    for b in all_batch_idx.unique():
        acc_b = all_acc[all_batch_idx == b]
        acc = acc_b[0, 0]
        all_acc_on_pedes.append(acc.unsqueeze(0))

    all_acc_on_pedes = torch.cat(all_acc_on_pedes, dim=0)
    return all_acc_on_pedes, (XX, YY)


def confidence_ellipse(x, y, loc_x, loc_y, ax, n_std=3.0, facecolor='none', cov=None, swap_scale_x_and_y=True, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of data *x* and *y*, at location *loc_x*, *loc_y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    loc_x, loc_y : float with location where to plot the ellipse

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if cov is None:
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    if swap_scale_x_and_y:
        ell_radius_x, ell_radius_y = ell_radius_y, ell_radius_x
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    if swap_scale_x_and_y:
        scale_x, scale_y = scale_y, scale_x  # swap since the y and x coordinate are swapped for the map we have (x is horizontal)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(loc_x, loc_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse), cov


def get_auxiliary_data(use_initial_pos_as_feature=True):
    """
    returns array of shape (nodes, feat) with auxiliary data
    """

    data_aux = utils.get_exit_nodes_realcoor(1)[:, 0, :]
    data_aux[..., :4] = utils.to_model_internal_coordinates(data_aux[..., :4].unsqueeze(1), True,True)[:,0]

    data_aux = torch.cat([data_aux,
                          data_aux[..., :2] if use_initial_pos_as_feature else torch.zeros(data_aux.shape[0], 2)
                          ], dim=-1)

    return data_aux


def plot_density_single_plot_v2(vector_gt, vector_sampled,bins_gt=50, bins_sampled=50, ax=None, s=32,
                                color_gt='tab:red', color_sampled='tab:blue', marker_gt=r'$\odot$',
                                linestyle_sampled='-', gt_label='Observed', sampled_label='Simulated', **hist_kwargs):

    if ax is None:
        ax = plt.gca()

    if vector_sampled is not None:
        y, binedges = np.histogram(vector_sampled, bins=bins_sampled, density=True, **hist_kwargs)
        bincenters = (binedges[1:] + binedges[:-1]) * 0.5
        ax.plot(bincenters, y, linestyle_sampled, color=color_sampled, label=sampled_label, markerfacecolor='none', zorder=-1,
                 linewidth=2.5)

    if vector_gt is not None:
        y, binedges = np.histogram(vector_gt, bins=bins_gt, density=True, **hist_kwargs)
        bincenters = (binedges[1:] + binedges[:-1]) * 0.5
        ax.scatter(bincenters, y, marker=marker_gt, color=color_gt, label=gt_label,
                    facecolor='none', zorder=1, s=s)




def plot_densities_subplots(pos_gt, pos_sample, vel_gt, vel_sample, cfg, prepend_to_title, append_to_title,
                            extension='.png'):
    # used for logging to wandb only
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False, sharex=False, figsize=(6.4, 4.8))

    # plot the position distributions as histograms:
    ax1.hist(pos_gt[..., 0], bins=50, density=True, color='tab:blue', label='ground-truth',
             histtype=u'step')
    ax1.hist(pos_sample[..., 0], bins=50, density=True, color='tab:orange', label='model samples',
             histtype=u'step')

    ax1.legend()
    ax1.set_xlabel(f'x position [mm]')# -- ks stat: {np.round(ksstat, 3)}, p-value: {np.round(ksp, 3)}')
    ax2.hist(pos_gt[..., 1], bins=50, density=True, color='tab:blue', label='ground-truth',
             histtype=u'step')
    ax2.hist(pos_sample[..., 1], bins=50, density=True, color='tab:orange', label='model samples',
             histtype=u'step')
    ax2.legend()
    ax2.set_xlabel(f'y position [mm]') # -- ks stat: {np.round(ksstat, 3)}, p-value: {np.round(ksp, 3)}')

    # plot the velocity distributions:
    low = -500
    high = 500
    ax3.hist(vel_gt[..., 0], bins=np.linspace(low, high, num=100), density=True, color='tab:blue', label='ground-truth',
             histtype=u'step')
    ax3.hist(vel_sample[..., 0], bins=np.linspace(low, high, num=100), density=True, color='tab:orange',
             label='model samples',
             histtype=u'step')
    ax3.legend()
    ax3.set_xlabel(f'x velocity [cm/s]') # -- ks stat: {np.round(ksstat, 3)}, p-value: {np.round(ksp, 3)}')

    low = -500
    high = 500

    ax4.hist(vel_gt[..., 1], bins=np.linspace(low, high, num=100), density=True, color='tab:blue', label='ground-truth',
             histtype=u'step')
    ax4.hist(vel_sample[..., 1], bins=np.linspace(low, high, num=100), density=True, color='tab:orange',
             label='model samples',
             histtype=u'step')
    ax4.legend()
    ax4.set_xlabel(f'y velocity [cm/s]') # -- ks stat: {np.round(ksstat, 3)}, p-value: {np.round(ksp, 3)}')

    fname = os.path.join(f"pedes_hists", f"{cfg['experiment']['state_dict_fname'][:-3]}")
    fname = os.path.join('figures', prepend_to_title, fname, append_to_title)
    if not os.path.exists(fname):
        os.makedirs(fname)

    plt.savefig(os.path.join(fname, 'hists' + extension), bbox_inches='tight')



def make_pretty_pdf(ax, vector_gt, vector_sample, bins=50, s=16,
                    color_sample='tab:blue', linestyle_sample='-',label_sample='simulated',
                    **kwargs):
    if vector_gt is not None:
        y, binedges = np.histogram(vector_gt, bins=bins // 2, density=True, **kwargs)
        bincenters = (binedges[1:] + binedges[:-1]) * 0.5
        ax.scatter(bincenters[::2], y[::2], marker=r'$\odot$', s=s, color='tab:red', facecolor='none', zorder=2,
                   label='observed')

    y, binedges = np.histogram(vector_sample, bins=bins, density=True, **kwargs)
    bincenters = (binedges[1:] + binedges[:-1]) * 0.5
    ax.plot(bincenters, y, linestyle_sample, color=color_sample, markerfacecolor='none', zorder=-1,
                  linewidth=2, label=label_sample)

def social_forcing_acc(pedes, neighbors, R=2., U0=0.375, top_ks=tuple(), max_angle=np.pi):
    # units in meters
    # pedes: (n_pedes, 2)
    # neighbors: (n_neighbors, 2)
    # formula for the social force in direction x: U0 * x/(R*dist) * exp(-dist / R) -- eq. 13 in Helbing 1995
    delta_coors = pedes.unsqueeze(1) - neighbors.unsqueeze(0)  # shape: (n_pedes, n_neighbors, 2)
    # delta_coors[i,j] gives the relative coordinate pos(i) - pos(j)
    dist = torch.norm(delta_coors, dim=-1, keepdim=True)  # shape: (n_pedes, n_neighbors, 1)
    EPS = 1e-6
    dist = torch.clamp(dist, min=EPS)
    # magnitude * direction
    force = (U0 / R * torch.exp(-dist / R)) * delta_coors / dist  # shape(n_pedes, n_neighbors, 2)

    # filter the forces s.t. we put those where delta_coors exceed max_angle to 0:
    angles = torch.atan2(delta_coors[..., 0], delta_coors[..., 1])
    mask_angle = (torch.abs(angles) > max_angle)  # & ((2*np.pi - torch.abs(angles)) > max_angle)
    force[mask_angle] = 0.

    # get the top-k forces:
    magnitudes = torch.norm(force, dim=-1)  # shape: (n_pedes, n_neighbors)
    top_k_forces = {}
    for k in top_ks:
        k_this = min(k, force.shape[1])
        topk_indices = torch.topk(magnitudes, k=k_this, dim=1, largest=True, sorted=False).indices
        force_topk = torch.zeros_like(force)
        force_topk[torch.arange(force.shape[0]), topk_indices] = force[torch.arange(force.shape[0]), topk_indices]
        assert force_topk.shape == force.shape, 'broadcasting error?'
        top_k_forces[k] = force_topk


    return force, top_k_forces

# Function to add equidistant markers to each streamline
def add_markers_to_streamline(ax, lines, marker, marker_interval=8, markersize=16, color='tab:red', alpha=0.7):
    from scipy.interpolate import interp1d
    segments = lines.get_segments()
    dists = []
    all_x = []
    all_y = []

    for i, segment in enumerate(segments):
        x, y = zip(*segment)
        all_x.append(x[0])
        all_y.append(y[0])
        # Compute the cumulative distance along the streamline
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        dists.append(dist)

    cumulative_dist = np.cumsum(np.concatenate(dists))
    # Create an interpolator to get points at regular intervals
    distance_new = np.arange(cumulative_dist[0], cumulative_dist[-1], marker_interval)
    interp_x = interp1d(cumulative_dist, all_x, kind='linear')
    interp_y = interp1d(cumulative_dist, all_y, kind='linear')

    # Interpolated points
    x_new = interp_x(distance_new)
    y_new = interp_y(distance_new)

    # Plot the interpolated points with markers
    ax.plot(x_new, y_new, linestyle='', marker=marker, markersize=markersize, color=color, alpha=alpha)

def make_hist_from_part_of_heatmap(ax, grouped_gt, grouped_sample, bin_id, hist_bins=80, new_ax=None,
                                   color_sample='tab:blue', linestyle_sample='-',label_sample='simulated'):

    sub_gt = grouped_gt.get_group(list(grouped_gt.groups.keys())[bin_id]) if grouped_gt is not None else None
    sub_sample = grouped_sample.get_group(list(grouped_sample.groups.keys())[bin_id])
    width = 397.48499  # width in points

    make_pretty_pdf(new_ax, sub_gt['x'] if sub_gt is not None else None,
                    sub_sample['x'], bins=hist_bins, s=32,
                    color_sample=color_sample, linestyle_sample=linestyle_sample,label_sample=label_sample)
    new_ax.set_yscale('log')
    # new_ax.set_xlabel('x [m]')
    # new_ax.set_ylabel('Conditional PDF')

    # plot the bin edges:
    if sub_gt is not None:
        ax.vlines(sub_gt['y'].min(), -10., 4., color='black', linestyle=':', alpha=0.9, zorder=10, linewidth=5.)
        ax.vlines(sub_gt['y'].max(), -10., 4., color='black', linestyle=':', alpha=0.9, zorder=10, linewidth=5.)


