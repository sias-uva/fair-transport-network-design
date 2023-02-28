import csv
import datetime
import json
from environment import Environment
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.optim as optim
from actor import DRL4Metro
from constraints import Constraints
from critic import StateCritic
import constants
from reward import ggi, group_utility, od_utility, discounted_development_utility, lowest_quintile_utility
import matplotlib.pyplot as plt
from mlflow import log_metric, log_artifact, log_param
from utils import gini

device = constants.device


def update_dynamic(dynamic, current_selected_index):
    """Updates the dynamic representation of the actor to a sparse matrix with all so-far selected stations.
    Note: this does not seem to be correct. The current implementation of metro expansion does not have any 'dynamic' elements, like demand.

    Args:
        dynamic (torch.Tensor): the current dynamic matrix.
        current_selected_index (np.int64): the latest selected station.

    Returns:
        torch.Tensor: the new dynamic matrix, where all selected stations are assigned 1.
    """
    dynamic = dynamic.clone()
    dynamic[0, 0, current_selected_index] = float(1)

    return dynamic


class Trainer(object):
    """Responsible for the wholet raining process."""

    def __init__(self, environment: Environment, constraints: Constraints, args):
        super(Trainer, self).__init__()

        # Prepare the models
        self.environment = environment
        self.actor = DRL4Metro(args.static_size,
                          args.dynamic_size,
                          args.hidden_size,
                          args.num_layers,
                          args.dropout,
                          update_dynamic,
                          environment.update_mask,
                          v_to_g_fn=environment.vector_to_grid,
                          vector_allow_fn=constraints.allowed_vector_indices).to(device)

        self.critic = StateCritic(args.static_size, args.dynamic_size,
                             args.hidden_size, environment.grid_size).to(device)

    def gen_line_plot_grid(self, lines):
        """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
        Essentially creates a grid of the given line to plot later on.

        Args:
            line (list): list of generated lines of the model
            grid_x_max (int): nr of lines in the grid
            grid_y_mask (int): nr of columns in the grid
        """
        data = np.zeros((self.environment.grid_x_size, self.environment.grid_y_size))

        for line in lines:
            line_g = self.environment.vector_to_grid(line)

            for i in range(line_g.shape[1]):
                data[line_g[0, i], line_g[1, i]] += 1
        
        data = data/len(lines)

        return data

    def train(self, args):
        """Performs the training over batches and epochs.

        Args:
            args (argparse.Namespace): parsed console arguments.
        """
        if args.checkpoint:
            self.actor.load_state_dict(torch.load(Path(args.checkpoint, 'actor.pt'), device))
            self.critic.load_state_dict(torch.load(Path(args.checkpoint, 'critic.pt'), device))

        now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
        save_dir = Path('./result') / f'{args.environment}_{now}'

        train_start = time.time()
        print(f'Starts training on {device} - Model location is {save_dir}')
        
        if not args.no_log:
            log_param('save_dir', save_dir)

            checkpoint_dir = save_dir / 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            with open(save_dir / 'args.txt', 'w') as f:
                # log the number of existing lines as a parameter.
                args_dict = vars(args)
                args_dict['existing_lines'] = len(self.environment.existing_lines)
                json.dump(args_dict, f, indent=2) 

        actor_optim = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        critic_optim = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        average_reward_list, actor_loss_list, critic_loss_list, average_od_list, average_Ac_list = [], [], [], [], []
        # best_params = None
        best_reward = 0

        static = self.environment.static
        dynamic = torch.zeros((1, args.dynamic_size, self.environment.grid_size),
                            device=device).float()  # size with batch

        for epoch in range(args.epoch_max):
            self.actor.train()
            self.critic.train()

            epoch_start = time.time()
            od_list, social_equity_list = [], []
            actor_loss = critic_loss = rewards_sum = 0

            for _ in range(args.train_size):  # this loop accumulates a batch
                tour_idx, tour_logp = self.actor(static, dynamic, args.station_num_lim, budget=args.budget,
                                            line_unit_price=args.line_unit_price, station_price=args.station_price,
                                            decoder_input=None, last_hh=None)

                reward = od_utility(tour_idx, self.environment)
                od_list.append(reward.item())
                ses_reward = torch.zeros(1)
                if args.reward == 'weighted':
                    # only calculate ses_reward if necessary
                    if args.ses_weight > 0:
                        ses_reward = discounted_development_utility(tour_idx, self.environment)
                        reward = args.ses_weight * ses_reward + (1-args.ses_weight) * reward
                elif args.reward == 'group':
                    reward = group_utility(tour_idx, self.environment, args.var_lambda, use_pct=not args.use_abs)
                # elif args.reward == 'group_weighted':
                #     reward = group_weighted_utility(tour_idx, self.environment, args.var_lambda, use_pct=not args.use_abs)
                elif args.reward == 'ai_economist':
                    reward = group_utility(tour_idx, self.environment, mult_gini=True, use_pct=not args.use_abs)
                elif args.reward == 'rawls':
                    # TODO CHANGE
                    # I messed up with the arguments and gave the dutch/western group as first and the non-western as second,
                    # so for this particular change the group we care about is the one with index = 1. But this needs to change in general.
                    group_idx = 0
                    if args.group_weights_files:
                        if len(args.group_weights_files) > 0:
                            group_idx = 1
                    
                    reward = lowest_quintile_utility(tour_idx, self.environment, use_pct=not args.use_abs, group_idx=group_idx)
                elif args.reward == 'ggi':
                    reward = ggi(tour_idx, self.environment, args.ggi_weight, use_pct=not args.use_abs)

                social_equity_list.append(ses_reward.item())

                critic_est = self.critic(static, dynamic, args.hidden_size,
                                    self.environment.grid_x_size, self.environment.grid_y_size).view(-1)
                advantage = (reward - critic_est)
                per_actor_loss = -advantage.detach() * tour_logp.sum(dim=1)
                per_critic_loss = advantage ** 2

                actor_loss += per_actor_loss
                critic_loss += per_critic_loss
                rewards_sum += reward

            actor_loss = actor_loss / args.train_size
            critic_loss = critic_loss / args.train_size
            avg_reward = rewards_sum / args.train_size
            average_od = sum(od_list)/len(od_list)
            average_Ac = sum(social_equity_list)/len(social_equity_list)

            average_reward_list.append(avg_reward.half().item())
            actor_loss_list.append(actor_loss.half().item())
            critic_loss_list.append(critic_loss.half().item())
            average_od_list.append(average_od)
            average_Ac_list.append(average_Ac)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            critic_optim.step()

            cost_time = time.time() - epoch_start
            print('epoch %d, average_reward: %2.3f, actor_loss: %2.4f,  critic_loss: %2.4f, cost_time: %2.4fs'
                % (epoch, avg_reward.item(), actor_loss.item(), critic_loss.item(), cost_time))


            torch.cuda.empty_cache()  # reduce memory
            
            if not args.no_log:
                log_metric('average_reward', avg_reward.item())
                log_metric('actor_loss', actor_loss.item())
                log_metric('critic_loss', critic_loss.item())
                log_metric('average_od', average_od)
                log_metric('average_ac', average_Ac)

                # Save the weights of an epoch
                epoch_dir = checkpoint_dir / str(epoch)
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)

                torch.save(self.actor.state_dict(), epoch_dir / 'actor.pt')
                torch.save(self.critic.state_dict(), epoch_dir / 'critic.pt')

                # Save best model parameters
                if avg_reward.item() > best_reward:
                    best_reward = avg_reward.item()

                    torch.save(self.actor.state_dict(), save_dir / 'actor.pt')
                    torch.save(self.critic.state_dict(), save_dir / 'critic.pt')

        if not args.no_log:
            with open(save_dir / 'reward_actloss_criloss.txt', 'w') as f:
                for i in range(args.epoch_max):
                    per_average_reward_record = average_reward_list[i]
                    per_actor_loss_record = actor_loss_list[i]
                    per_critic_loss_record = critic_loss_list[i]
                    per_epoch_od = average_od_list[i]
                    per_epoch_Ac = average_Ac_list[i]

                    to_write = f'{per_average_reward_record}\t{per_actor_loss_record}\t{per_critic_loss_record}\t{per_epoch_od}\t{per_epoch_Ac}\n'

                    f.write(to_write)

        plt.plot(average_reward_list, '-', label="reward")
        plt.title(f'Reward vs. epochs - {now}')
        plt.ylabel('Reward')
        plt.legend(loc='best')
        if not args.no_log:
            plt.savefig(save_dir / 'loss.png', dpi=800)
            log_artifact(save_dir / 'loss.png')

        print(f'Finished training in {(time.time() - train_start)/60} minutes.')
        if not args.no_log:
            log_metric('training_time', (time.time() - train_start)/60)

    def evaluate(self, args):
        assert args.result_path, 'args.checkpoint folder needs to be given to evalute a model'

        self.actor.eval()

        # Load the models.
        self.actor.load_state_dict(torch.load(Path(args.result_path, 'actor.pt'), device))
        self.critic.load_state_dict(torch.load(Path(args.result_path, 'critic.pt'), device))

        # Setup the initial static and dynamic states.
        static = self.environment.static
        dynamic = torch.zeros((1, args.dynamic_size, self.environment.grid_size),
                            device=device).float()  # size with batch

        # generate 128 different lines to have a bigger sample size
        gen_lines = []
        for _ in range(args.train_size):
            with torch.no_grad():
                tour_idx, _ = self.actor(static, dynamic, args.station_num_lim, decoder_input=None, last_hh=None)
                gen_lines.append(tour_idx)

        if not args.no_log:
            with open(Path(args.result_path, 'tour_idx_multiple.txt'), "w", newline='') as f:
                wr = csv.writer(f)
                wr.writerows([line[0].tolist() for line in gen_lines])

        # Plot the average generated line (from the multiple sample generated lines)
        plot_grid = self.gen_line_plot_grid(gen_lines)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(plot_grid)
        fig.suptitle(f'{args.environment} - Average Generated line \n from {args.result_path}')
        fig.savefig(Path(args.result_path, 'average_generated_line.png'))
        # log_artifact(Path(args.result_path, 'average_generated_line.png'))
        # create a list of pairs of indices for the average generated line, to save in the results.
        nz = plot_grid.nonzero()
        avg_gen_line = np.stack((nz[0], nz[1]), axis=-1)

        # Evaluate metrics
        satisfied_ods = np.zeros(len(gen_lines))
        satisfied_group_ods = np.zeros((len(gen_lines), len(self.environment.group_od_mx))) # make an array of dimensions lines x groups to store ods by line by group
        # avg distance of every square to the nearest line stop.
        distances = np.zeros(len(gen_lines))
        group_distances = np.zeros((len(gen_lines), len(self.environment.group_od_mx)))
        for i, line in enumerate(gen_lines):
            # Evaluate ODs
            sat_od_mask = self.environment.satisfied_od_mask(line)
            satisfied_ods[i] = (sat_od_mask * self.environment.od_mx).sum().item()

            # Evaluate average distance to nearest public transport station.
            # Manhattan distance between each grid cell and the average generated line.
            # line_g = self.environment.vector_to_grid(line).transpose(0, 1)
            # Calculate the distance from every grid cell to every station in the generated line.
            # dist = torch.cdist(self.environment.grid_indices.float().to(device), line_g.float(), p=1)
            # Find the distance of the station closest to each cell. Reshape as the grid (grid_x x grid_y)
            # min_dist = dist.min(axis=1)[0].reshape_as(self.environment.grid_groups)
            # average distance of all grids with an assigned group (not NaN)
            # distances[i] = (~self.environment.grid_groups.isnan() * min_dist).sum() / (~self.environment.grid_groups.isnan()).sum()

            if self.environment.group_od_mx:
                for j, g_od in enumerate(self.environment.group_od_mx):
                    satisfied_group_ods[i, j] = (g_od * sat_od_mask).sum().item()

                    # group_distances[i, j] = ((self.environment.grid_groups == self.environment.groups[j]) * min_dist).sum() / (self.environment.grid_groups == self.environment.groups[j]).sum()

        mean_sat_od = satisfied_ods.mean()
        mean_sat_od_pct = mean_sat_od / (self.environment.od_mx.sum() / 2)
        mean_sat_od_by_group = satisfied_group_ods.mean(axis=0)
        mean_sat_od_by_group_pct = mean_sat_od_by_group / [g.cpu().sum()/2 for g in self.environment.group_od_mx]
        total_group_od = sum([g.sum()/2 for g in self.environment.group_od_mx])
        mean_distance = distances.mean()
        mean_group_distance = group_distances.mean(axis=0)

        # Plot bars of satisfied ODs by group and overall
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].bar(range(mean_sat_od_by_group.shape[0]), mean_sat_od_by_group)
        axs[0].title.set_text(f'Mean Satisfied OD by group \n Total OD: {round(mean_sat_od, 2)} - Total group OD: {round(mean_sat_od_by_group.sum(), 2)} \n Model: {args.result_path}')
        axs[1].bar(range(mean_sat_od_by_group.shape[0]), mean_sat_od_by_group_pct)
        axs[1].title.set_text(f'Mean Satisfied OD % by group \n Total OD: {mean_sat_od_pct} - Total group OD: {sum(mean_sat_od_by_group)/total_group_od}  \n Model: {args.result_path}')

        fig.savefig(Path(args.result_path, 'satisfied_od_by_group.png'))
        # log_artifact(Path(args.result_path, 'satisfied_od_by_group.png'))

        # Plot bars of average distance to nearest stop by group and overall
        # fig, ax = plt.subplots(figsize=(7, 5))
        # ax.bar(range(mean_group_distance.shape[0]), mean_group_distance)
        # ax.title.set_text(f'Mean distance to nearest stop by group \n Mean Overall Distance: {round(mean_distance, 2)} \n Model: {args.result_path}')
        # ax.axhline(y=mean_distance,linewidth=1, color='gray', ls='--')

        # fig.savefig(Path(args.result_path, 'mean_distance_to_stop.png'))

        group_gini = gini(mean_sat_od_by_group)
        group_pct_gini = gini(mean_sat_od_by_group_pct)

        # Create .json file with all result metrics.
        result_metrics = {
            'avg_generated_line': avg_gen_line.tolist(),
            'mean_sat_od': mean_sat_od,
            'mean_sat_od_pct': mean_sat_od_pct.item(),
            'mean_sat_od_by_group': mean_sat_od_by_group.tolist(),
            'mean_sat_od_by_group_pct': mean_sat_od_by_group_pct.tolist(),
            'mean_sat_group_od': sum(mean_sat_od_by_group).item(),
            'mean_sat_group_od_pct': (sum(mean_sat_od_by_group)/total_group_od).item(),
            'group_gini': group_gini,
            'group_pct_gini': group_pct_gini,
            'mean_distance': mean_distance,
            'mean_group_distance': mean_group_distance.tolist()
        }
        
        if not args.no_log:
            with open(Path(args.result_path, 'result_metrics.json'), 'w') as outfile:
                json.dump(result_metrics, outfile)
