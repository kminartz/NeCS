import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils
from modules.engine.Enc_Proc_Dec import Enc_Proc_Dec
from shapely import Polygon
import torch_scatter as ts


class NeCS(nn.Module):

    def __init__(self, num_layers, data_dim=10, emb_dim=16, nonlinearity=nn.GELU(), dynamic_channels=(0,1,2,3),
                 num_latent_dim=0, use_fixed_std=False,
                 pred_stepsize=1,
                 dist_thresh=5.,
                 sample_from_pred=False, type_channels=(4, 5, 6, 7),
                 recurrent_unit=nn.LSTM,
                 apply_edge_correction_step=True, rel_pos_loss_coef=1.,
                 scale_decoder_sigma=1.,
                 col_push_strength=0, col_push_R=1.,
                 **kwargs):

        super().__init__()

        """
        NeCS model for crowd simulations in Eindhoven train station
        :param num_layers: number of layers in the GNN models
        :param data_dim: dimension of the input data (incl augmentations such as type indicator channels)
        :param emb_dim: dimension of the latent embeddings
        :param nonlinearity: activation function
        :param dynamic_channels: channels that are updated by the model (as opposed to kept constant)
        :param num_latent_dim: number of latent dimensions in the VAE
        :param use_fixed_std: whether to use a fixed std for the VAE, s.t. the reconstruction loss is equivalent to MSE
        :param pred_stepsize: number of 0.1s steps to predict into the future
        :param dist_thresh: threshold for the distance between nodes to create an edge
        :param sample_from_pred: whether to sample from the prediction distribution or use the mean
        :param type_channels: channels that indicate the type of the node (pedestrian, exit, geometry, flowing) -- note: geometry is not used, but the indicator channel is kept for compatibility with the model dict
        :param recurrent_unit: RNN unit to use for the memory states
        :param apply_edge_correction_step: whether to apply an edge correction step to the predicted positions
        :param rel_pos_loss_coef: coefficient for the weighting of the relative position loss
        :param scale_decoder_sigma: scaling factor for the predicted sigma
        :param col_push_strength: strength of the social collision avoidance push force
        :param col_push_R: radius of the collision push force
        :param kwargs: additional parameters for the vae, gnn, etc.    
        """

        self.im_dim = data_dim
        all_channels = (i for i in range(data_dim))
        self.dynamic_channels = dynamic_channels if dynamic_channels is not None else all_channels
        self.static_channels = tuple(set(all_channels) - set(dynamic_channels))

        self.use_fixed_std = use_fixed_std
        self.type_channels = type_channels

        # mappers from embedding space to mu and sigma of the output distribution
        self.emb_to_mu = nn.Sequential(
            nonlinearity,
            nn.Linear(emb_dim, emb_dim),  # we overwrite the static channels later
            nonlinearity,
            nn.Linear(emb_dim, len(self.static_channels) + len(self.dynamic_channels))
        )
        self.emb_to_std = nn.Sequential(
            nonlinearity,
            nn.Linear(emb_dim, emb_dim),  # we overwrite the static channels later
            nonlinearity,
            nn.Linear(emb_dim, len(self.static_channels) + len(self.dynamic_channels))
        )

        self.scale_vel_for_update = nn.Linear(emb_dim, 2)

        # vae-gnn backbone:
        self.engine = Enc_Proc_Dec(num_layers, data_dim, emb_dim, nonlinearity=nonlinearity,
                                   dynamic_channels=dynamic_channels, num_latent_dim=num_latent_dim, **kwargs)
        # other info:
        self.dist_thresh = dist_thresh
        self.kwargs = kwargs
        self.sample_from_pred = sample_from_pred
        self.idx_pos = (0, 1)
        self.idx_vel = (2, 3)
        self.pred_stepsize = pred_stepsize
        self.rel_pos_loss_coef = rel_pos_loss_coef
        self.scale_decoder_sigma = scale_decoder_sigma
        self.col_push_strength = col_push_strength
        self.col_push_R = col_push_R

        # recurrent unit init:
        self.recurrent_unit = None
        if recurrent_unit is not None and str(recurrent_unit).lower() != 'none':
            self.recurrent_unit = recurrent_unit(emb_dim, emb_dim, batch_first=True)
        print('using recurrent unit: ', self.recurrent_unit)

        # edge correction coefficient network init:
        self.apply_edge_correction_step = apply_edge_correction_step
        if apply_edge_correction_step:
            self.edge_corr_coef_net = nn.Sequential(nn.Linear(emb_dim, 2), nn.Tanh())


    def forward(self, x: torch.Tensor, x_true: torch.Tensor = None, batch_idx=None, *memory_terms):
        EPS = 1e-2
        if len(memory_terms) == 0:  # first pass in autoregressive chain
            memory_terms = None

        # type tensor: 1 -> pedestrian, 2 -> exit
        type = self._get_type_tensor(x)
        assert torch.logical_or(type == 1, type == 2).all(), 'expected only type 1 and type 2 for pedes and exit nodes!'

        # first get some general info about the batch:
        pedes_mask = x[..., 4] == 1
        flowing_mask = x[..., 7] == 1

        # make the edges:
        edge_index = self.construct_edge_indices(x, batch_idx, type, dist_thresh=self.dist_thresh)
        edge_attr = self.construct_edge_features(x, edge_index)

        input_model = (x, edge_index, edge_attr, type)

        # if we have x_true (= x^{t+1}), we also give the gt node features as input to the model:
        # note: no edge indices or feat, we only use the input edge idx or feat
        if x_true is not None:
            edge_index_true = torch.Tensor([[], []]).to(edge_index)
            edge_attr_true = self.construct_edge_features(x_true, edge_index)
            input_model_gt = (x_true, edge_index_true, edge_attr_true, type)
        else:
            input_model_gt = None

        # now we can process the input with the VAE backbone:
        engine_output, additional_loss = self.engine(input_model, input_model_gt)
        assert isinstance(engine_output, tuple), 'engine output should be a tuple of (node_feats, edge_index, edge_attr)'
        edge_engine_output = engine_output[2]  # the embedded edge features

        engine_output = engine_output[0]  # node embeddings -- update with recurrent unit:
        engine_output, memory_terms = self.update_memory_states(engine_output, memory_terms=memory_terms, type=type)

        # get those edges that are between pedestrians:
        pedes_edge_mask = torch.bitwise_and(type[edge_index[0]] == 1, type[edge_index[1]] == 1)
        edge_index_pedes = edge_index[:, pedes_edge_mask]

        # initialize the parameters of the output distr. over positions and velocities
        mu = torch.zeros_like(x)
        log_std = torch.zeros_like(x)

        # map the engine output from abstract embedding space to data space:
        out_mu = self.emb_to_mu(engine_output)
        out_log_sigma = self.emb_to_std(engine_output)

        # first update the velocity:
        mu[..., self.idx_vel] = x[..., self.idx_vel] + out_mu[..., self.idx_vel]
        log_std[..., self.idx_vel] = out_log_sigma[..., self.idx_vel]

        # use the updated velocity to update the position with euler:
        mu[..., self.idx_pos] = x[..., self.idx_pos]
        # converts to real coordinate system, and then does euler step, then reverts to model internal system
        mu[..., (*self.idx_pos, *self.idx_vel)] = utils.model_internal_euler_step(
            mu[..., (*self.idx_pos, *self.idx_vel)], self.pred_stepsize
        )

        if self.apply_edge_correction_step:
            # apply edge correction step to positions
            # get learned coefficients:
            edge_eng_output_pedes = edge_engine_output[pedes_edge_mask]
            edge_coefs = self.edge_corr_coef_net(edge_eng_output_pedes)
            # calculate 1/(1+dist^2):
            pairwise_diff_vec = (mu[edge_index_pedes[0]] - mu[edge_index_pedes[1]])[..., self.idx_pos]
            pairwise_diff_vec = pairwise_diff_vec / (
                    1 + torch.linalg.norm(pairwise_diff_vec, dim=-1, keepdim=True) ** 2
            )
            pairwise_diff_vec = pairwise_diff_vec * (1 - torch.isnan(pairwise_diff_vec).float())  # set nan to 0
            # apply correction to each node:
            edge_corrs = pairwise_diff_vec * edge_coefs.unsqueeze(1)
            edge_corr = ts.scatter(src=edge_corrs, index=edge_index_pedes[0], dim=0, dim_size=x.shape[0], reduce='sum')
            mu[..., self.idx_pos] = mu[..., self.idx_pos] + edge_corr

        if self.col_push_strength > 0:
            # apply a manual collision avoidance force:
            dist = torch.linalg.norm(pairwise_diff_vec, dim=-1, keepdim=True)  # dist in decameters
            forces = (self.col_push_strength / self.col_push_R * torch.exp(-(dist * 10) / self.col_push_R)  # dist * 10 to go to meters
                      ) * pairwise_diff_vec / dist
            force = ts.scatter(src=forces, index=edge_index_pedes[0], dim=0, dim_size=x.shape[0], reduce='sum')
            mu[..., self.idx_vel] = mu[..., self.idx_vel] + force  #update vel (in m/s)
            mu[..., self.idx_pos] = mu[..., self.idx_pos] + force / 10   #update pos (in decameters)

        log_std[..., self.idx_pos] = out_log_sigma[..., self.idx_pos]

        ## postprocessing of the output:
        sigma = (nn.Softplus()(log_std) + EPS) * self.scale_decoder_sigma
        if self.use_fixed_std:  # use std so that optimizing reconstr loss is equivalent to mse
            sigma[..., (*self.idx_pos, *self.idx_vel)] = torch.ones_like(sigma[..., (*self.idx_pos, *self.idx_vel)]) * (
                0.5) ** 0.5

        # now set the static channels (type indicators, init pos)
        mu[..., self.static_channels] = x[..., self.static_channels]
        sigma[..., self.static_channels] = 1 / ((2 * torch.pi) ** 0.5)  # set to value s.t. log_prob is 0 here

        # keep non-pedestrian nodes fixed:
        mu[~pedes_mask] = x[~pedes_mask]
        sigma[~pedes_mask] = 1 / ((2 * torch.pi) ** 0.5)  # set to value s.t. log_prob is 0 here

        # detach non-flowing nodes as they will be updated in postprocessing step AFTER the backward pass,
        # so that the gradients are not propagated to them:
        mu[~flowing_mask] = mu[~flowing_mask].detach()
        sigma[~flowing_mask] = sigma[~flowing_mask].detach()

        # calc relpos loss:
        relpos_loss = self.calc_rel_pos_loss(mu, x_true, edge_index_pedes)

        # construct decoding distribution:
        pred_dist = torch.distributions.Normal(mu, sigma)

        # get sample, keeping into account fixed static channels and non-pedes nodes:
        pred_sample = pred_dist.rsample() if self.sample_from_pred else pred_dist.mean
        pred_sample[..., self.static_channels] = x[..., self.static_channels]  # keep static channels fixed
        pred_sample[~pedes_mask] = x[~pedes_mask]

        add_loss_dict = {'KL': additional_loss, 'relpos': relpos_loss}


        return pred_dist, add_loss_dict, pred_sample, batch_idx, *memory_terms

    @torch.no_grad()
    def get_additional_val_stats(self, pred_sample, x_true):

        # calculate sum of euclidean distance between pred and x_true:
        pred_sample_real = torch.ones_like(pred_sample) * pred_sample
        x_true_real = torch.ones_like(x_true) * x_true
        pred_sample_real[..., :4] = utils.to_real_coordinates(pred_sample[..., :4], True, True)
        x_true_real[..., :4] = utils.to_real_coordinates(x_true[..., :4], True,True)

        return torch.linalg.norm(pred_sample_real[..., self.idx_pos] - x_true_real[..., self.idx_pos], dim=(1, 2)).sum()

    def construct_edge_indices(self, x, batch_idx, type, dist_thresh=5):

        edge_index = []
        node_counter = 0
        for b in batch_idx.unique():
            one_batch = x[batch_idx == b]
            type_batch = type[batch_idx == b]
            edge_idx_within_batch = self.get_edge_index_for_one_batch(one_batch, type_batch, dist_thresh=dist_thresh)
            edge_idx = edge_idx_within_batch + node_counter  # (2, n_edges)
            edge_index.append(edge_idx)
            node_counter += one_batch.shape[0]
        edge_index = torch.cat(edge_index, dim=-1)
        self._validate_edge_index(x, edge_index, batch_idx, type)
        return edge_index

    def _validate_edge_index(self, x, edge_index, batch_idx, type):
        # unit test to check if there are no indices between batches
        row, col = edge_index
        batch_idx_row = batch_idx[row]
        batch_idx_col = batch_idx[col]
        assert torch.all(
            batch_idx_row == batch_idx_col), 'edge index contains edges between nodes from different batch elements!'

        # we expect exactly one edge from each pedes to a global node and vice versa:
        to_pedes = (type[row] == 1)
        to_global = (type[row] == 0)
        from_pedes = (type[col] == 1)
        from_global = (type[col] == 0)
        num_pedes = torch.sum(type == 1)

        if torch.sum(type == 0) > 0:  # we have global nodes
            assert torch.sum(torch.bitwise_and(to_pedes, from_global)) == num_pedes, 'expected exactly one edge to each pedes from a global node!'
            assert torch.sum(torch.bitwise_and(to_global, from_pedes)) == num_pedes, 'expected exactly one edge from each pedes to a global node!'

        # check if connected pedes are at most self.dist_thresh meters away from each other:
        x_real_coor = utils.to_real_coordinates(x[...,:4], True,True)
        edge_ped_ped = edge_index[:, torch.bitwise_and(to_pedes, from_pedes)]
        dist = torch.linalg.norm(x_real_coor[edge_ped_ped[0]][...,:2] - x_real_coor[edge_ped_ped[1]][..., :2], dim=-1)
        assert torch.all(dist <= self.dist_thresh * 1000 + 1),\
            f'expected all connected pedes to be at most {self.dist_thresh} meters away from each other!, but got max distance of {torch.max(dist)}'
        # *1000 to go to mm

        return

    @torch.no_grad()
    def get_edge_index_for_one_batch(self, x, type_batch, dist_thresh=1.0):

        # shape of x: (batch*nodes, feats)
        # dist between nodes
        dist = self.construct_distmat(x)
        adj_mat = dist <= dist_thresh
        # add edges between exit nodes and all pedestrian nodes:
        exit_nodes = (type_batch == 2).long()
        pedestrian_nodes = (type_batch == 1).long()
        geom_nodes = (type_batch == 3).long()
        non_global = (type_batch != 0).long()
        exit_pedestr_adj_mat = torch.einsum('i,j->ij', exit_nodes, pedestrian_nodes).bool().transpose(0, 1)
        adj_mat += exit_pedestr_adj_mat
        adj_mat = adj_mat * pedestrian_nodes.unsqueeze(
            1)  # only pedestrians have incoming edges (aggregation happens at row indicies, first row of edge_index)
        geom_geom_adj_mat = torch.einsum('i,j->ij', geom_nodes, geom_nodes).bool().transpose(0, 1)
        adj_mat += torch.bitwise_and(geom_geom_adj_mat,
                                     dist <= dist_thresh)  # geometry nodes have incoming edges from other geometry nodes, but not from pedestrian nodes
        adj_mat = torch.bitwise_and(adj_mat,
                                    torch.einsum('i,j->ij', non_global, non_global)
                                    )  # only non-global edges until now - global added in next step
        adj_mat.fill_diagonal_(0)  # no self loops

        edge_index = torch.nonzero(adj_mat).transpose(0, 1)  # (2, N)

        return edge_index

    def construct_edge_features(self, x, edge_idx: torch.Tensor):
        if x is None:
            return None
        # let's have relative velocity magnitude, relative distance as input for the edges

        row, col = edge_idx
        edge_feat = torch.zeros(size=(row.shape[0], 2 + len(self.type_channels) ** 2))
        source = x[row]
        target = x[col]
        dist = torch.linalg.norm(source[..., self.idx_pos] - target[..., self.idx_pos], dim=(1, 2))
        vel_diff = torch.linalg.norm(source[..., self.idx_vel] - target[..., self.idx_vel], dim=(1, 2))

        edge_feat[..., 0] = dist
        edge_feat[..., 1] = vel_diff
        # one hot encoding of edge type:
        edge_feat[..., 2:] = torch.einsum('ijk,ijl->ijkl', source[..., self.type_channels],
                                          target[..., self.type_channels]).reshape(-1, len(self.type_channels) ** 2)

        return edge_feat.to(
            x)

    @torch.no_grad()
    def construct_distmat(self, x, use_real_coors=True):
        if x is None:
            return None

        if use_real_coors:
            x = utils.to_real_coordinates(x[..., :4],True, True) / 1000.  # convert to meters

        x = x[..., :2]

        # let's have relative velocity magnitude, relative distance as input for the edges
        coors_rep = x.repeat(1, x.size(0), 1)
        dist_squared = torch.sum((coors_rep - coors_rep.transpose(0, 1)) ** 2, dim=-1)

        diagonal_mask = torch.eye(dist_squared.shape[0], dist_squared.shape[1]).to(dist_squared.device)
        dist = torch.sqrt(
            dist_squared + diagonal_mask * 1e-6)  # the 1e-6 here prevents NaN gradient error for 0 under sqrt
        dist = dist * (1 - diagonal_mask)
        return dist

    def update_memory_states(self, engine_output, memory_terms=None, type=None):
        """
        update the memory states of the model by processing the engine output, when applicable
        NOTE: this operates node-wise on the engine output to preserve permutation equivariance

        Returns (output, (h_n, c_n)), where output is the updated engine output, which took into account the memory states
        this output should then be mapped to the real data space by the map_to_... layers. h_n and c_n should be passed
        along the autoregressive chain for the next time step.
        """



        if self.recurrent_unit is None:
            return engine_output, tuple()

        assert len(engine_output.shape) == 3 and engine_output.shape[1] == 1, \
            'engine output should be of shape (bs, 1, feats)'

        assert self.recurrent_unit.batch_first, 'recurrent unit should be batch first!'

        # give engine_output as input to the recurrent unit with memory:
        if memory_terms is None:
            updated_engine_output, memory_terms_updated = self.recurrent_unit(engine_output)  # h_{t+1}, same for c
        else:
            if type is None:
                type = torch.ones_like(engine_output[..., 0, 0])

            memory_terms_extended = tuple(
                torch.zeros(
                    (engine_output.shape[0], engine_output.shape[1], memory_terms[i].shape[-1])
                ).to(engine_output) for i in range(len(memory_terms))
            )
            for i, m in enumerate(memory_terms_extended):
                m[type != 0] = memory_terms[i]
                assert (m[type == 0] == 0).all(), 'expected no memory for global nodes!'

            memory_terms_perm = (torch.permute(m, dims=[1,0,2]) for m in memory_terms_extended)  # go back to batch on second dim
            # fill any NaN values with 0s -- these are the newly added nodes which have not yet had their memory component initialized
            memory_terms_perm = tuple(torch.nan_to_num(m, nan=0.) for m in memory_terms_perm)
            updated_engine_output, memory_terms_updated = self.recurrent_unit(engine_output,
                                                                        memory_terms_perm)  # h_{t+1}, same for c

        if not isinstance(memory_terms_updated, tuple):
            memory_terms_updated = (memory_terms_updated,)

        # put batch on first dim (index 0)
        assert updated_engine_output.shape[1] == 1 and all((m.shape[0] == 1 for m in memory_terms_updated)), \
            'expected seq length dimension to be 1 for recurrent unit output and expected one layer only for one directional hidden dim!'

        return updated_engine_output, tuple(torch.permute(m, dims=[1,0,2]) for m in memory_terms_updated)


    def calc_rel_pos_loss(self, pred_sample, x_true, edge_index):
        """
        calculate the relative position vector difference between true and prediction
        """
        # calculate additional relative position loss to encourage good relative positioning:
        if self.rel_pos_loss_coef > 0. and x_true is not None:
            pairwise_diff_vec_true = (x_true[edge_index[0]] - x_true[edge_index[1]])[..., self.idx_pos]
            pairwise_diff_vec_pred = (pred_sample[edge_index[0]] - pred_sample[edge_index[1]])[..., self.idx_pos]
            return torch.linalg.norm(pairwise_diff_vec_pred - pairwise_diff_vec_true, dim=-1).sum() * self.rel_pos_loss_coef
        else:
            return torch.tensor([0.]).to(pred_sample)

    def _get_type_tensor(self, x):
        type = torch.zeros_like(x[:, 0, 4])
        type[x[:, 0, 4] == 1] = 1 # pedes
        type[x[:, 0, 5] == 1] = 2 # exit
        type[x[:, 0, 6] == 1] = 3 # geometry nodes (not used)
        type[x[:, 0, 7] == 1] = 1 # pedes

        # in earlier versions, we experimented with so-called geometry nodes and global nodes, but these are not used:
        assert (type == 3).sum() == 0, 'expected no geometry nodes in the type tensor!'
        assert (type == 0).sum() == 0, 'expected no global nodes in the type tensor!'

        return type


