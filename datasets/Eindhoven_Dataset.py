import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import utils
import matplotlib.pyplot as plt

g = torch.Generator()
g.manual_seed(0)

def get_data_loaders(config, limit_num_data_points_to=np.inf, **kwargs):
    # get the dataloaders
    data_directory, batch_size = config['data_directory'], config['batch_size']
    use_vavg = config['use_vavg']

    dir_train = os.path.join(data_directory, 'train')
    dir_val = os.path.join(data_directory, 'val')
    dir_test = os.path.join(data_directory, 'test')

    dataset = EhvDataset(dir_train, num_samples=limit_num_data_points_to,
                         use_vavg=use_vavg, **kwargs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph,
                                             generator=g, worker_init_fn=seed_worker)


    val_dataset = EhvDataset(dir_val, use_vavg=use_vavg, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                 collate_fn=collate_graph,
                                                 generator=g, worker_init_fn=seed_worker)

    test_dataset = EhvDataset(dir_test, use_vavg=use_vavg, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                  collate_fn=collate_graph,
                                                  generator=g, worker_init_fn=seed_worker)

    print('Number of train datapoints:', len(dataset.fnames),
          'number of val datapoints:', len(val_dataset.fnames), 'number of test datapoints:',
          len(test_dataset.fnames)
          )

    return dataloader, val_dataloader, test_dataloader

class EhvDataset(Dataset):

    def __init__(self, dir_path, num_samples=np.inf,
                 distinguish_flowing=True,
                 use_initial_pos_as_feature=True,
                 pred_stepsize=10, use_vavg=True):


        self.dir = dir_path
        all_fnames = os.listdir(dir_path)
        num_samples = min(num_samples, len(all_fnames))
        self.fnames = all_fnames[:num_samples]

        example = np.load(os.path.join(dir_path, self.fnames[0]))  # get an example datapoint
        if not hasattr(example, 'shape'):  # compressed npz file -- access 'data' key
            example = example['data']
            # shape: (num_bodies, attributes, time), but we permute the dims later to (time, num_bodies, attributes)

        self.num_timesteps = example.shape[2]
        self.samples_per_epoch_multiplier = len(all_fnames) / len(self.fnames)  # keep definition of epoch consistent if we restrict to less training data

        # some flags that represent preprocessing/augmentation choices
        self.distinguish_flowing = distinguish_flowing  # indicate whether pedestrians flowing towards the exit (yes)
        self.use_initial_pos_as_feature = use_initial_pos_as_feature  # indicate whether the initial position where someone appeared is a feature (yes)
        self.use_vavg = use_vavg  # indicate whether instantaneous vel or vel avgd over 1 second is used as feature (latter)

        self.pred_stepsize = pred_stepsize
        assert pred_stepsize == 10, \
            'please double check if you intend to change the prediction stepsize and if there could be any incompatibilities in the code for other values.'

        # load the data into RAM and preprocess:
        self.data = dict()
        for i, fname in enumerate(self.fnames):  # preprocessing
            data_np = np.load(os.path.join(self.dir, fname))
            self.data[i] = self.load_and_preprocess(data_np)


    def __getitem__(self, idx):
        return self.data[idx]

    def load_and_preprocess(self, data_np):

        np_datapoint = data_np
        if not hasattr(np_datapoint, 'shape'):  # compressed npz file
            np_datapoint = np_datapoint['data']


        datapoint: torch.Tensor = torch.from_numpy(np_datapoint).float()

        if self.use_vavg:
            # do not use instantaneous velocity, but overwrite the velocity with the avgd vel:
            datapoint = self._get_and_overwrite_vavg(datapoint)

        #  add type indicators to pedestrian nodes (type indicator is [1,0,0, flowing={0,1}])
        # --> flowing will be concatenated later
        type_pedestrian = torch.Tensor([1,0,0]).repeat(datapoint.shape[0], datapoint.shape[1], 1).to(datapoint)
        datapoint = torch.cat([datapoint, type_pedestrian], dim=-1)  # all data so far is pedestrian data

        if self.distinguish_flowing:
            # now append the 'flowing' vs 'non-flowing' type indicator, which distinguishes between pedestrians that we
            # want to model (those that are flowing towards the exit) vs those we do not want to model (those that are non-flowing)
            flowing = self._get_flowing_vs_non_flowing_type(datapoint)  # (people, time, 1)
        else:
            # assume all pedestrian nodes are flowing
            flowing = torch.ones(datapoint.shape[0], datapoint.shape[1], 1).to(datapoint)

        datapoint = torch.cat([datapoint, flowing], dim=-1)

        # now add the exit nodes: (these have a type indicator of [0,1,0,0])
        exit_nodes = utils.get_exit_nodes_realcoor(t=datapoint.shape[1]).to(datapoint)
        datapoint = torch.cat([datapoint, exit_nodes], dim=0)

        # note: we used to experiment with 'geometry nodes', which had a type indicator of [0,0,1,0], but did not keep these
        # still, the indicator channel is kept to ensure compatibility with the model weights used for the paper. This
        # does not affect the method since these nodes do not exist, and the third value is always 0.

        # put nans everywhere if the pedestrian is not present since these will anyway be filtered out and so this will pass all checks
        datapoint[torch.isnan(datapoint[..., 0])] = torch.nan

        # Normalize data to reasonable range by converting the units so that they are in reasonable range
        # (decameter for pos, m/s for vel)
        datapoint[..., :4] = utils.to_model_internal_coordinates(datapoint[..., :4], True,
                                                                 use_torch=True)

        if self.use_initial_pos_as_feature:
            # add the initial position as a feature to the node
            # the initial position of a node is the first non-NaN entry along the time dimension, which is dim 1

            # first, find the first non-NaN entry along the time dimension
            first_non_nan_idx = torch.argmax(
                torch.ones_like(datapoint[..., 0]) * ~torch.isnan(datapoint[..., 0]), dim=1
            )  # note: argmax returns first occurence
            # now, get the initial position of each node
            initial_position = datapoint[torch.arange(datapoint.shape[0]), first_non_nan_idx, :2]
            # now, repeat the initial position along the time dimension
            initial_position = initial_position.unsqueeze(1).repeat(1, datapoint.shape[1], 1)  # shape: num_people, time, 2
        else:
            # simply assume initial pos to be (0,0) for all nodes
            initial_position = torch.zeros(datapoint.shape[0], datapoint.shape[1], 2).to(datapoint)

        datapoint = torch.cat([datapoint, initial_position], dim=-1)

        # finally, replace the additional features with NaN where the node is not observed for consistency purposes
        datapoint[torch.isnan(datapoint[..., 0])] = torch.nan

        return datapoint.unsqueeze(1)

    def __len__(self):
        return len(self.fnames)

    def _get_flowing_vs_non_flowing_type(self, data):
        # we make a distinction between flowing and non-flowing pedestrians
        # flowing pedestrians are those that are moving towards the exit
        # a pedestrian is flowing if:
        # - the smallest y coordinate (horizontal in the image of the paper) is at least 15000 mm smaller than the largest y coordinate
        # - the time at which the pedestrian is at the smallest y coordinate is later than the time of the largest y-coordinate

        # data has shape (people, time, feat)

        # first, get the smallest and largest y coordinate for each pedestrian, as well as the times
        min_y, t_min_y = torch.min(torch.nan_to_num(data[..., 1], torch.inf), dim=1)
        max_y, t_max_y = torch.max(torch.nan_to_num(data[..., 1], -1*torch.inf), dim=1)

        # now, get the flowing pedestrians
        flowing = torch.bitwise_and(min_y < max_y - 15_000, t_min_y > t_max_y)

        # expand the flowing vector along the time dimension, as add a dimension so it can easily be concatenated with the data
        flowing = flowing.unsqueeze(1).repeat(1, data.shape[1]).unsqueeze(-1)

        # put NaNs where the pedestrian is not observed:
        flowing[torch.isnan(data[..., 0])] = torch.nan

        return flowing  # (people, time 1)


    def _get_and_overwrite_vavg(self, x):
        # assert that we only have pos and vel info at this time:
        assert x.shape[-1] == 4, 'expected only pos and vel info here!'
        # get the average velocity of each pedestrian, averaged over the last pred_stepsize timesteps
        pos = x[:, :-self.pred_stepsize, :2]  # pos up to last prediction step
        pos_shifted=  x[:, self.pred_stepsize:, :2]  # pos after first prediction step
        diff = pos_shifted - pos   #(p, t, 2)  # difference in positions one prediction step apart (in mm)
        vel = diff / (self.pred_stepsize) # in mm / 0.1 s = cm/s

        x_new = torch.zeros_like(x[:, self.pred_stepsize:])
        x_new[...,:2] = x[:, self.pred_stepsize:, :2]
        x_new[...,2:4] = vel
        nan_feat = torch.isnan(x_new).any(dim=(2))
        x_new[nan_feat] = torch.nan
        return x_new


def collate_graph(list_of_tensors):  # list of tensors contains tuples (x,y)
    x_out = []
    batch_idx_out = []
    for id, (x) in enumerate(list_of_tensors):
        batch_idx_out.append(torch.Tensor([id for _ in range(x.shape[0])]))
        x_out.append(x)

    x_out = torch.cat(x_out, dim=0)
    batch_idx_out = torch.cat(batch_idx_out, dim=0)
    return x_out.float(), batch_idx_out.long()



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

