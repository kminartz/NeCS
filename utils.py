import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.nn.functional as F
import warnings
import matplotlib.image as mpimg


class Config():
    def __init__(self, config_dict):
        self.config_dict = config_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(cfg_str, state_dict=None):
    import configs
    cfg = getattr(configs, cfg_str)
    cfg_dict = dict(cfg.config_dict)

    if state_dict is not None:  # replace state dict str with new state dict
        old = cfg_dict['experiment']['state_dict_fname']
        new_state_dict_pointer = {'state_dict_fname': state_dict}
        cfg_dict['experiment'] = new_state_dict_pointer
        print(f'changed state dict from {old} \t\t\t to {state_dict} \t\t\t in {cfg_str}!')

    cfg = Config(config_dict=cfg_dict)
    return cfg


def make_dict_serializable(config):
    import json
    for k, v in config.items():
        try:
            json.dumps(v)
        except:
            config[k] = str(v)
    return config


def set_size_plots(width, fraction=1, h_to_w_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.
    source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27


    if h_to_w_ratio is None:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2
        ratio = golden_ratio
    else:
        ratio = h_to_w_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def filter_input(x_prelim: torch.Tensor, y_prelim: torch.Tensor, other_prelim_gt):
    """
    filter out the nodes that did not yet exist in the last timestep, as well as the ones that are currently NaN.
    return the tensor y as well as the idx of the non-nan nodes in y and the idx of the nodes in y that are also in x

    This makes the implicit assumption that nodes that do no longer exist in the next timestep will not affect the
    currently existing nodes!
    """

    idx_in_x = get_non_nan_idx(x_prelim)

    if y_prelim is not None:
        assert x_prelim.shape[0] == y_prelim.shape[0], 'expected full tensors including NaNs as input!'

        idx_to_pred = get_non_nan_idx(y_prelim)
        # now, remove those idx from idx_to_pred that do not yet appear in x, since we cannot predict those:
        idx_to_keep = torch.from_numpy(
            np.intersect1d(idx_in_x.cpu(), idx_to_pred.cpu())
        ).to(idx_in_x.device)
        y = y_prelim[idx_to_keep]
    else:
        idx_to_keep = idx_in_x
        y = None

    x = x_prelim[idx_to_keep]

    miscellaneous = []
    for i, _ in enumerate(other_prelim_gt):
        try:
            m = other_prelim_gt[i][idx_to_keep]
            miscellaneous.append(m)
        except:
            warnings.warn(
                f'could not filter from {i}th element in miscellaneous list! please check if this is expected behavior!')

    assert (x.shape[0] == y.shape[0]) or (y is None), 'expected equal number of nodes in x and y!'
    # x (filtered), y (filtered), idx of nodes in x, y, idx of non-nan in y_prelim, idx of nodes in y
    return x, y, idx_to_keep, idx_to_pred, miscellaneous



def construct_new_x(y_prelim_gt, y_pred_sample, idx_to_keep, idx_to_pred, detach_grad_chain=True, distinguish_non_flowing=True,
                    other_prelim_gt=tuple(), other_sample=tuple()):
    # now, the model input for the next steps becomes y_pred
    x = torch.zeros_like(y_prelim_gt)  # initialize tensor with 0s which has room for NaNs
    x[:] = torch.nan
    if detach_grad_chain:
        y_pred_sample = y_pred_sample.detach()
    x[idx_to_keep] = y_pred_sample  # put predictions for nodes that were already present and remain present

    # now, for newly appearing nodes in the next timestep y, always put the ground truth:
    mask_idx_to_add = torch.isin(idx_to_pred, idx_to_keep, assume_unique=True,
                             invert=True)  # true if the idx_to_pred is NOT in idx_to_keep
    idx_to_add = idx_to_pred[mask_idx_to_add]
    x[idx_to_add] = y_prelim_gt[idx_to_add]

    if distinguish_non_flowing:
        # for nodes that are non-flowing, put the ground-truth
        # distinguish between nodes that are not flowing and nodes that are flowing
        flowing = x[..., 7]
        assert torch.bitwise_or(torch.bitwise_or(flowing == 0, flowing == 1), torch.isnan(flowing)).all(), 'expected only 0 and 1 in flowing channel!'
        # now set all non-flowing nodes to the ground truth of the next timestep:
        x[flowing == 0] = y_prelim_gt[flowing == 0]

    assert (torch.isnan(x) == torch.isnan(y_prelim_gt)).all(), 'expected same NaNs in gt y and rollout prediction'

    miscellaneous = []
    for i, o in enumerate(other_sample):
        misc_i = torch.zeros(y_prelim_gt.shape[0], *o.shape[1:]).to(other_sample[i]) * torch.nan
        if detach_grad_chain:
            o = o.detach()
        misc_i[idx_to_keep] = o.to(misc_i)
        if i < len(other_prelim_gt):
            misc_i[idx_to_add] = other_prelim_gt[i][idx_to_add].to(misc_i)  # initialize the gt misc features for newly added nodes
        miscellaneous.append(misc_i)

    return x, miscellaneous


def _get_cafe_polygon(model_internal_coors=False, in_mm=False):
    from shapely import Polygon
    if in_mm:
        cafe_corners = [[-500, 28000], [-700, 50000], [-5200, 50000], [-5000, 28000]]
    elif not model_internal_coors:
        cafe_corners = [[-0.5, 28], [-0.7, 50], [-5.2, 50], [-5., 28]]  # in meters, absolute values
    else:
        cafe_corners = [[-4.05, -1.2], [-4.07, 1.0], [-4.52, 1.0], [-4.5, -1.2]]  # in decameters, shifted coordinate system for model
    poly = Polygon(cafe_corners)
    return poly


def get_non_nan_idx(x: torch.Tensor):
    dims_to_sum_over = tuple(i for i in range(1, len(x.shape)))
    mask = torch.isnan(x).sum(dims_to_sum_over) <= 0
    idx_to_keep = torch.nonzero(mask)[:,0]
    return idx_to_keep


def plot_station_background(ax=None, divide_extent_by=1, bg=None):

    if ax is None:
        ax = plt.gca()

    total_width = 18000
    configuration_simple = {
        "background_parameters": {
            "map/station_map.pdf": {
                "x_min": -2500,
                "y_min": -8950 - (total_width-12500) / 2,  # -11700
                "x_max": 75000,
                "y_max": 3450 + (total_width-12500) / 2  # 6200
            }
        }
    }

    bg_params = configuration_simple["background_parameters"]["map/station_map.pdf"]
    bg = mpimg.imread('map/station_map.png') if bg is None else bg
    ax.imshow(bg, extent=[
        bg_params["x_min"] / divide_extent_by, bg_params["x_max"] / divide_extent_by,
        bg_params["y_min"] / divide_extent_by, bg_params["y_max"] / divide_extent_by,
    ], alpha=1)

    ax.set_xlim(-5000 / divide_extent_by, 76000 / divide_extent_by)
    ax.set_ylim(-13000 / divide_extent_by, 7000 / divide_extent_by)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    return bg_params




def to_model_internal_coordinates(data, to_model_internal_coors: bool, use_torch: bool):
    """
    convert the coordinates from the data to the model internal coordinates
    x coor is approximately in range from -8000mm to 4000mm
    y coor is approximately in [0, 75000]
    vx is approx in [-100,100]
    vy is approx in [-200, 200]
    We normalize this s.t. everything is in reasonable numerical range
    Parameters
    ----------
    data: arr with shape [num_people, ..., *feats], where the first four feats are x, y, vx, and vy respectively

    Returns: arr with normalized coordinates for internal model processing
    -------

    """

    if to_model_internal_coors:
        if not use_torch:
            data_out = np.zeros_like(data)
        else:
            data_out = torch.zeros_like(data)
        data_out[..., :2] = data[..., :2] / 10_000 - 4
        data_out[..., 0] = data_out[..., 0] + 4.3  # shift x coordinate to the right by 4.3 decameters so that it is +- centered around 0
        data_out[..., 2:4] = data[..., 2:4] / 100
        return data_out

    raise NotImplementedError('to_model_internal_coors should be true')


def to_real_coordinates(normalized_data, from_model_internal_coors: bool, use_torch: bool):
    """
    inverts the normalization of to_model_internal_coordinates -- see docs there
    Parameters
    ----------
    data

    Returns
    -------

    """
    assert normalized_data.shape[-1] == 4, 'expected last dimension to be 4 (x,y,vx,vy)!'

    if from_model_internal_coors:
        if not use_torch:
            data_out = np.zeros_like(normalized_data)
        else:
            data_out = torch.zeros_like(normalized_data)
        data_out[..., 0] = normalized_data[..., 0] - 4.3  # shift x coordinate to the right by 4.3 decameters so that it is centered around 0
        data_out[..., 1] = normalized_data[..., 1]
        data_out[..., :2] = (data_out[..., :2] + 4) * 10_000
        data_out[..., 2:4] = normalized_data[..., 2:4] * 100
        return data_out
    else:
        # go to real coordinates based on min-max normalized data in [-1,1]
        if use_torch:
            min = torch.zeros_like(normalized_data[..., :])
            max = torch.zeros_like(normalized_data[..., :])
        else:
            min = np.zeros_like(normalized_data[..., :])
            max = np.zeros_like(normalized_data[..., :])

        min[..., 0] = -8000
        max[..., 0] = 4000
        min[..., 1] = 0
        max[..., 1] = 75000
        min[..., 2] = -100
        max[..., 2] = 100
        min[..., 3] = -200
        max[..., 3] = 200
        return inverse_min_max_norm(normalized_data, min, max)


def min_max_norm(x, min, max, upper=1, lower=-1):
    return (x - min) / (max - min) * (upper - lower) + lower

def inverse_min_max_norm(x, min, max, upper=1, lower=-1):
    return (x - lower) / (upper - lower) * (max - min) + min


def convert_internal_velocity_to_internal_coor_update(pos_and_vel_data, dt, preserve_coordinates=True, use_torch=True):
    """
    convert the velocity update to a position update in internal coordinates. Euler integration step on top of current pos
    Parameters
    ----------
    pos_and_vel_data: arr with shape [num_people, ..., *feats], where the first four feats are x, y, vx, and vy respectively
    dt: time step size

    Returns: arr with shape [num_people, ..., *feats], where the first four feats are x, y, vx, and vy respectively
    -------

    """
    data_unnormalized = to_real_coordinates(pos_and_vel_data, use_torch=use_torch, from_model_internal_coors=preserve_coordinates)
    assert torch.allclose(pos_and_vel_data,
                          to_model_internal_coordinates(data_unnormalized, use_torch=use_torch, to_model_internal_coors=preserve_coordinates),
                          atol=1e-4, rtol=1e-4), 'expected to get the same coordinates back after converting to model internal coordinates and back!'
    # here v is in cm/s, x in millimeters, so we need to multiply v by dt (dt = pred_stepsize in 0.1s units) to get the position update in mm
    data_unnormalized[..., :2] = data_unnormalized[..., :2] + data_unnormalized[..., 2:4] * dt
    data_normalized = to_model_internal_coordinates(data_unnormalized, use_torch=use_torch, to_model_internal_coors=preserve_coordinates)
    return data_normalized

def model_internal_euler_step(pos_and_vel_data, dt, use_torch=True):
    """
    convert the velocity update to a position update in internal coordinates. Euler integration step on top of current pos
    Parameters
    ----------
    pos_and_vel_data: arr with shape [num_people, ..., *feats], where the first four feats are x, y, vx, and vy respectively
    dt: time step size

    Returns: arr with shape [num_people, ..., *feats], where the first four feats are x, y, vx, and vy respectively
    -------

    """
    data_unnormalized = to_real_coordinates(pos_and_vel_data, from_model_internal_coors=True, use_torch=use_torch)
    assert torch.allclose(pos_and_vel_data,
                          to_model_internal_coordinates(data_unnormalized, to_model_internal_coors=True, use_torch=use_torch),
                          atol=1e-4, rtol=1e-4), 'expected to get the same coordinates back after converting to model internal coordinates and back!'
    # here v is in cm/s == mm/ds, x in millimeters, so we need to multiply v by dt (dt = pred_stepsize in 0.1s units) to get the position update in mm
    data_unnormalized[..., :2] = data_unnormalized[..., :2] + data_unnormalized[..., 2:4] * dt
    data_normalized = to_model_internal_coordinates(data_unnormalized, to_model_internal_coors=True, use_torch=use_torch)
    return data_normalized


def _check_if_inside_cafe(pos, slack, model_internal_coors=False, in_mm=False):
    # check if pos is inside the cafe:
    cafe = _get_cafe_polygon(model_internal_coors, in_mm)  # convert to millimeters
    corners = cafe.exterior.coords.xy
    corners = torch.from_numpy(np.asarray(corners)).to(pos)
    # approximate by rectangle by taking the min/max x and y coordinates:
    x_min = torch.min(corners[0]) - slack
    x_max = torch.max(corners[0]) + slack
    y_min = torch.min(corners[1]) - slack
    y_max = torch.max(corners[1]) + slack
    # check if pos is inside the rectangle:
    inside_rectangle = torch.bitwise_and(pos[..., 0] >= x_min, pos[..., 0] <= x_max)
    inside_rectangle = torch.bitwise_and(inside_rectangle, pos[..., 1] >= y_min)
    inside_rectangle = torch.bitwise_and(inside_rectangle, pos[..., 1] <= y_max)

    return inside_rectangle, (x_min, x_max, y_min, y_max)


def get_exit_nodes_realcoor(t=589):
    exit_nodes = torch.Tensor([[500, 2000, 0, 0, 0, 1, 0, 0],
                               [-600, 2000, 0, 0, 0, 1, 0, 0],
                               [-1700, 2000, 0, 0, 0, 1, 0, 0],
                               [-2800, 2000, 0, 0, 0, 1, 0, 0],
                               [-3900, 2000, 0, 0, 0, 1, 0, 0],
                               [-5000, 2000, 0, 0, 0, 1, 0, 0]]
                              ).unsqueeze(1).repeat(1, t, 1)

    return exit_nodes


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def gaussian_pdf(x, mu=0, sigma=1):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
