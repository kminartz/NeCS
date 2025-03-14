import random
import os
import warnings
import pickle
import torch
import torch.nn as nn
import time

import utils
from utils import filter_input, get_non_nan_idx, construct_new_x
from evaluation.experiment_utils import model_rollout
import numpy as np
import matplotlib.pyplot as plt
from utils import *
try:
    import wandb
    use_wandb=True
except:
    use_wandb=False
    # print('wandb not found!')
import evaluation.experiment_functions as ef

class Trainer():

    def __init__(self, model: nn.Module, loss_func, opt, pred_steps, num_kl_annealing_cycles=1,
                 kl_increase_proportion_per_cycle=1, config=None, try_use_wand=True, beta=1,
                 clip_reconstr_loss_to=torch.inf, detach_bptt_grad_chain=True,
                 increase_rollout_epochs=tuple(50 + i*10 for i in range(100)), clip_grad_norm=None,
                 full_config=None):


        # store the model etc:
        self.model = model
        self.loss_func = loss_func
        self.pred_stepsize = pred_steps
        self.opt: torch.optim.Optimizer = opt
        self.num_kl_annealing_cycles = num_kl_annealing_cycles
        self.kl_increase_proportion_per_cycle = kl_increase_proportion_per_cycle
        self.use_wandb = use_wandb if try_use_wand else False
        self.beta = beta
        self.reconstr_loss_clip = clip_reconstr_loss_to
        self.detach_bptt_grad_chain = detach_bptt_grad_chain
        self.increase_rollout_epochs = increase_rollout_epochs
        self.clip_grad_norm = clip_grad_norm if clip_grad_norm is not None else torch.inf
        self.config = full_config

        if self.use_wandb:
            proj_name = 'NeCS'
            try:
              wandb.init(project=proj_name, config=utils.make_dict_serializable(config))
              wandb.watch(self.model, log='all', log_freq=100)
            except Exception as e:
              warnings.warn('failed to connect to WandB, will proceed without it. Exception:')
              print(e)
              self.use_wandb = False


    def train(self, loader, val_loader, epochs, device, training_strategy='simple',
              save_fname=None, start_from_epoch=0):

        self.device = device
        if save_fname is None:
            save_fname = 'models/state_dicts/last_experiment.pt'
            print(f'saving weights to default path {save_fname}')

        # save some stats:
        train_losses = []
        train_acc = []
        val_losses = []
        val_acc = []
        val_rollout_losses = []
        val_rollout_acc = []
        best_state_dict = None
        last_state_dict = None
        best_rollout_loss = torch.inf

        print(f'using {training_strategy} training!')

        for ep in range(start_from_epoch, epochs):
            start_ep = time.time()
            # get beta for this epoch, run the epoch:
            beta = self._get_beta_kl_annealing_schedule(ep, epochs)
            loss, mean_metric, mean_additional_loss_dict = self.train_one_ep(loader, device, training_strategy, ep, beta)

            train_losses.append(loss), train_acc.append(mean_metric)
            val_loss = 0
            val_mean_metric = 0
            val_mean_additional_loss = {}
            if val_loader is not None:
                # do validation pass:
                val_loss, val_mean_metric, val_mean_additional_loss_dict = self.val(val_loader, device)
                val_losses.append(val_loss)
                val_acc.append(val_mean_metric)
            stop = time.time()
            print(
                f'epoch {ep} \t\t loss {loss:2f} \t\t additional loss terms {mean_additional_loss_dict} \t\t mean correct {mean_metric:2f}'
                f'\t\t '
                f'val loss {val_loss} \t\t val additional loss terms {val_mean_additional_loss_dict} \t\t val mean correct {val_mean_metric:2f}'
                f'\t\t took {stop - start_ep:2f} seconds'
            , flush=True)

            if self.use_wandb:
                # logging:
                wandb.log({'loss':loss, 'mean metric':mean_metric, **mean_additional_loss_dict,
                           'reconstruction loss':loss - mean_additional_loss_dict['KL'],
                           'val loss': val_loss, 'val mean metric': val_mean_metric, **val_mean_additional_loss_dict,
                           'val reconstruction loss': val_loss - val_mean_additional_loss_dict['val KL'],
                           'epoch': ep, 'beta': beta})

            if ep % 20 == 0 and val_loader is not None:  # every 10 epochs we do a more extensive validation:

                # first: log model weights:
                if not os.path.exists(os.path.join('models', 'state_dicts', 'last')):
                    os.makedirs(os.path.join('models', 'state_dicts', 'last'))

                last_state_dict = self.model.state_dict()
                torch.save(last_state_dict, os.path.join('models', 'state_dicts', 'last', save_fname))
                if self.use_wandb:  # log the state dict to WandB
                    art = wandb.Artifact(save_fname + 'last', type="model")
                    art.add_file(os.path.join('models', 'state_dicts', 'last', save_fname))
                    wandb.log_artifact(art)

                start_val_rollout = time.time()
                try:
                    # make a full rollout
                    val_rollout_loss, val_rollout_mean_metric = self.val_with_rollout(val_loader, device)
                    val_rollout_losses.append(val_rollout_loss)
                    val_rollout_acc.append(val_rollout_mean_metric)
                    print(
                        f'val rollout loss: {val_rollout_loss}, val rollout mean metric: {val_rollout_mean_metric}, took {time.time() - start_val_rollout:2f} seconds',
                        flush=True)

                    if val_rollout_loss <= best_rollout_loss:
                        # use MSE of the full rollout as proxy loss:
                        print(f'found new best weights at epoch {ep}')
                        best_rollout_loss = val_rollout_loss
                        best_state_dict = self.model.state_dict()
                        torch.save(best_state_dict, os.path.join('models', 'state_dicts', save_fname))
                        if self.use_wandb:
                            art = wandb.Artifact(save_fname + 'best', type="model")
                            art.add_file(os.path.join('models', 'state_dicts', save_fname))
                            wandb.log_artifact(art)
                except Exception as e:
                    print(f'!!!! failed to run full eval at epoch {ep}!')
                    print('exception:', e)


        return train_losses, train_acc, val_losses, val_acc, best_state_dict, last_state_dict

    def train_one_ep(self, loader, device, training_strategy, epoch, beta=1):

        self.model.to(device)
        self.model.train()

        mean_loss = 0
        mean_additional_loss = {}
        count = 0
        mean_metric = 0
        max_rollout_len_set_flag = False
        max_rollout_len_pfw = 1

        for (batch, data) in enumerate(loader):
            # iterate over the dataloader
            c = data.size(0) if not isinstance(data, tuple) else data[0].size(0)  # counter
            s = data.size(2) if not isinstance(data, tuple) else data[0].size(2)  # time series length in a datapoint
            max_rollout_len_pfw = self.get_rollout_len(epoch, (s - 1) // self.pred_stepsize)
            max_rollout_len_pfw = int(max_rollout_len_pfw)


            rollout_length = max_rollout_len_pfw

            if not max_rollout_len_set_flag:
                # log some scheduled values:
                max_rollout_len_set_flag = True
                print(f'max training rollout len in this epoch is {max_rollout_len_pfw}. Beta: {beta}')

            if training_strategy == 'bptt':
                loss, metric, additional_loss_dict = self.train_step_bptt(data, device, rollout_length, beta)
            else:  # we could implement other training strategies, but here only use bptt/multi-step training
                raise NotImplementedError()

            mean_loss += loss
            # store some other values for inspection (KL, rel pos loss,...)
            for k, v in additional_loss_dict.items():
                if k not in mean_additional_loss.keys():
                    mean_additional_loss[k] = 0
                mean_additional_loss[k] += v

            mean_metric += metric  # proxy metric
            count += c

        # calculate output data for logging:
        mean_loss = mean_loss / count
        mean_metric = mean_metric / count
        for k, v in mean_additional_loss.items():
            mean_additional_loss[k] = v / count

        return mean_loss, mean_metric, mean_additional_loss

    @torch.no_grad()
    def val(self, val_loader, device):
        self.model.eval()
        mean_loss = 0
        mean_additional_loss_dict = {}
        count = 0
        mean_metric = 0


        for data in val_loader:
            data, *miscellaneous_all = data
            batch_idx = miscellaneous_all[0] if len(miscellaneous_all) > 0 else None
            end_of_sim_time = data.size(2)

            # get the randomly sampled start times to start the prediction step
            if batch_idx is None:  # no batch idx, should not happen!
                raise ValueError('expected batch idx tensor')
            else:
                batch_idx_unique = torch.unique(batch_idx)
                num_times = batch_idx_unique.size(0)
                start_time_b = np.random.randint(low=0, high=end_of_sim_time - self.pred_stepsize,
                                                 size=num_times)
                start_time = start_time_b[batch_idx]
                assert (np.diff(start_time[
                                    batch_idx == 0]) == 0).all(), 'expected same start time for all nodes in the first trajectory!'

            assert start_time.shape[0] == data.size(0), 'expected same number of start times as batch size!'

            target_time = start_time + self.pred_stepsize

            # get input and target output
            x_prelim = data[range(data.size(0)), :, start_time].to(device)
            y_prelim = data[range(data.size(0)), :, target_time].to(device)

            # put batch_idx tensors etc on device
            for i, o in enumerate(miscellaneous_all):
                o = o.to(device)
                miscellaneous_all[i] = o

            # filter the input (e.g. remove NaNs correspoding to pedestrians not present at this time)
            x, y, idx_in_x, idx_to_pred, miscellaneous = filter_input(x_prelim, y_prelim, miscellaneous_all)

            # prediction:
            y_pred_dist, additional_loss_dict, y_pred_disc, *miscellaneous = self.model(x, y, *miscellaneous)

            # get and save losses:
            additional_loss = 0
            for k, v in additional_loss_dict.items():
                additional_loss = additional_loss + v
                if k not in mean_additional_loss_dict.keys():
                    mean_additional_loss_dict[k] = 0
                mean_additional_loss_dict[k] += v.item()

            loss = -y_pred_dist.log_prob(y).sum() + additional_loss
            mean_loss += loss.item()
            count += data.size(0)
            mean_metric += self.model.get_additional_val_stats(y_pred_disc, y)

        return mean_loss / count, mean_metric / count, {'val ' + k: v/count for k, v in mean_additional_loss_dict.items()}

    @torch.no_grad()
    def val_with_rollout(self, val_loader, device):
        self.model.eval()
        mean_loss = 0
        count = 0
        mean_metric = 0
        rollout_length_max = 100


        for data in val_loader:
            data, *miscellaneous_all = data
            data = data.to(device)
            start_time = 0
            rollout_length = (data.size(2) - 1) // self.pred_stepsize
            rollout_length = min(rollout_length_max, rollout_length)

            # start model rollout
            trues, pred, pred_cont = model_rollout(
                self.model, data, self.pred_stepsize, rollout_length, start_time, True,
                miscellaneous_all=miscellaneous_all)
            # output logistics to get everything in the right shape:
            trues_all = torch.cat([torch.from_numpy(arr) for arr in trues[1:]], dim=0)
            pred_all = torch.cat([torch.from_numpy(arr) for arr in pred[1:]], dim=0)
            pred_cont_all = torch.cat([torch.from_numpy(arr) for arr in pred_cont[1:]], dim=0)

            trues, pred_cont, idx_to_keep, idx_to_pred, _ = filter_input(trues_all, pred_cont_all,
                                                                         [torch.arange(0, trues_all.shape[0])]  # placeholder, is not relevant here
                                                                         )  # filtering NaNs etc
            # log proxy loss and metrics
            loss = torch.sum(self.loss_func(pred_cont, trues))
            mean_loss += loss.item()
            count += data.size(0)
            trues, pred, idx_to_keep, idx_to_pred, _ = filter_input(trues_all, pred_all,
                                                                    [torch.arange(0, trues_all.shape[0])]# placeholder, is not used internally
                                                                    )
            mean_metric += self.model.get_additional_val_stats(pred, trues)

        return mean_loss / count / rollout_length, mean_metric / count / rollout_length


    def train_step_bptt(self, data, device, rollout_length, beta):

        # notes:
        # The number of nodes in the data tensor will be constant, only the number of NaNs in each timepoint will vary
        # We filter out predictions for nodes that will become NaN in y
        # For the unrolling, at each time we add new nodes to x when a new node appears in the gt
        # We filter nodes from y that are NaN in x as these are yet to appear

        self.opt.zero_grad()

        # some trackers:
        loss_over_all_steps = 0
        add_loss_over_all_steps = {}
        add_stat_over_all_steps = 0
        data, *miscellaneous_all = data  # other can be batch idx etc, possibly empty list
        batch_idx = miscellaneous_all[0] if len(miscellaneous_all) > 0 else None

        # get initial conditions from random start time
        end_of_sim_time = data.size(2)
        if batch_idx is None:  # no batch idx, should not happen:
            raise ValueError('Expected batch idx tensor')
        else:
            batch_idx_unique = torch.unique(batch_idx)
            num_times = batch_idx_unique.size(0)
            start_time_b = np.random.randint(low=0, high=end_of_sim_time - self.pred_stepsize * rollout_length, size=num_times)
            start_time = start_time_b[batch_idx]
            assert (np.diff(start_time[batch_idx == 0]) == 0).all(), 'expected same start time for all nodes in the first trajectory!'
 
        assert start_time.shape[0] == data.size(0), 'expected same number of start times as batch size!'

        # input data and batch tensors:
        x_prelim = data[range(data.size(0)), :, start_time].to(device)
        for i, o in enumerate(miscellaneous_all):
            o = o.to(device)
            miscellaneous_all[i] = o

        # initialize miscellaneous object which will be updated depending on which nodes are present at each time -- miscellaneous all will serve as gt to extract data
        # from for newly appearing nodes
        miscellaneous = [torch.ones_like(m) * m for m in miscellaneous_all]

        # do multiple prediction steps with gradient tracking
        backprop_loss = 0

        for step in range(1, rollout_length+1):
            target_time = start_time + self.pred_stepsize * step
            y_prelim = data[range(data.size(0)), :, target_time].to(device)
            assert x_prelim.shape[0] == y_prelim.shape[0], 'expected same shapes for x and y before filtering!'

            # remove pedestrians not observed (NaN values) for this time:
            x, y, idx_to_keep, idx_to_pred, miscellaneous = filter_input(
                x_prelim, y_prelim, miscellaneous)

            # predict:
            y_pred_dist, additional_loss_dict, y_pred_disc, *miscellaneous = self.model(x, y, *miscellaneous)

            # loss calc:
            additional_loss = 0
            for k, v in additional_loss_dict.items():
                additional_loss = additional_loss + v

            loss = - torch.clip(
                y_pred_dist.log_prob(y), -self.reconstr_loss_clip, self.reconstr_loss_clip
            ).sum() + beta*additional_loss
            _loss = loss / y.shape[0]  # normalize to account for the varying # nodes per batch

            backprop_loss = backprop_loss + _loss / rollout_length
            if self.detach_bptt_grad_chain:
                # we could choose to detach the chain at each step and backprop only a single step,
                # a la Brandstetter 2022 "Message Passing Neural PDE solvers"
                backprop_loss.backward()
                backprop_loss = 0

            # the below two tensors are merely for tracking statistics
            loss_over_all_steps = loss_over_all_steps + loss.detach().item()
            for k, v in additional_loss_dict.items():
                if k not in add_loss_over_all_steps.keys():
                    add_loss_over_all_steps[k] = 0
                add_loss_over_all_steps[k] = add_loss_over_all_steps[k] + v.detach().item()

            with torch.no_grad():
                add_stat = self.model.get_additional_val_stats(y_pred_disc, y)  # some proxy metric indicating closeness
                add_stat_over_all_steps += add_stat

            # make the new model input:
            x_prelim, miscellaneous = construct_new_x(y_prelim, y_pred_disc, idx_to_keep, idx_to_pred,
                                       detach_grad_chain=self.detach_bptt_grad_chain, other_prelim_gt=miscellaneous_all,
                                       other_sample=miscellaneous)

        # normalize:
        loss_over_all_steps /= rollout_length
        for k, v in add_loss_over_all_steps.items():
            add_loss_over_all_steps[k] /= rollout_length
        add_stat_over_all_steps /= rollout_length

        # backprop if we did not detach the chain:
        if not self.detach_bptt_grad_chain:
            backprop_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.opt.step()

        return loss_over_all_steps, add_stat_over_all_steps, add_loss_over_all_steps

    def _get_beta_kl_annealing_schedule(self, current_ep, total_epochs):

        cycle_length = total_epochs // self.num_kl_annealing_cycles
        ep_in_this_cycle = current_ep % cycle_length
        p_of_cycle = (ep_in_this_cycle + 1) / cycle_length

        if current_ep >= cycle_length * self.num_kl_annealing_cycles:
            beta = self.beta
        elif p_of_cycle > self.kl_increase_proportion_per_cycle:
            beta = self.beta
        else:
            beta = min(p_of_cycle / self.kl_increase_proportion_per_cycle * self.beta, self.beta)

        return beta

    def get_rollout_len(self, ep, timesteps_per_sample):
        for i, e in enumerate(self.increase_rollout_epochs):
            if ep < e:
                return min(i+1, timesteps_per_sample)
        return min(len(self.increase_rollout_epochs) + 1, timesteps_per_sample)



