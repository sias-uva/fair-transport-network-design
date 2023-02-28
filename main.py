import argparse

from constraints import ForwardConstraints
from environment import Environment
from trainer import Trainer
from pathlib import Path
from mlflow import log_metric, log_param, log_artifacts

# torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Network Expansion with Reinforcement Learning")

    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--static_size', default=2, type=int)
    parser.add_argument('--dynamic_size', default=1, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--epoch_max', default=300, type=int)
    parser.add_argument('--train_size',default=128, type=int) # like a batch_size
    parser.add_argument('--line_unit_price', default=1.0, type=float)
    parser.add_argument('--station_price', default=5.0, type=float)
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--actor_lr', default=10e-4, type=float)
    parser.add_argument('--critic_lr', default=10e-4, type=float)
    parser.add_argument('--station_num_lim', default=45, type=int)  # limit the number of stations in a line
    parser.add_argument('--budget', default=210, type=int)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--environment', default='diagonal_5x5', type=str)
    parser.add_argument('--ignore_existing_lines', action='store_true', default=False) # if true, the agent will build a line from scratch, ignoring what already the city has
    # reward types:
        # - weighted: a weighted sum of OD and equity reward -> --ses_weight * r_ses + (1-ses_weight) * r_od
        # - group: ODs are measured/regularized by group (see --groups_file), not a single OD.
        # - ai_economist: reward used by the ai_economist paper: total_utility * (1-gini(total_utility))
        # - rawls: returns the total satisfied OD of the lowest quintile group
        # - ggi: generalized gini index over group -> weight controlled by --ggi_weight
        # - group_weighted: group based utility but each cell has a different weight for each group -> requires --group_weights_files
    parser.add_argument('--reward', default='weighted', type=str)
    parser.add_argument('--ses_weight', default=0, type=float) # weight to assign to the socio-economic status (equity)reward, only works for --reward=weighted
    parser.add_argument('--var_lambda', default=0, type=float) # weight to assign to the variance of the satisfied OD among groups, only works for --reward=group
    parser.add_argument('--ggi_weight', default=2, type=float) # weight to assign when calculating the ggi of the satisfied OD among groups, only works for --reward=ggi

    parser.add_argument('--groups_file', default=None, type=str) # file that contains group membership of each grid square (e.g. when each square belongs to a certain income bin).
    parser.add_argument('--group_weights_files', default=None, nargs="*") # files that contain group weights of each grid square (e.g. when each square has a percentage of a certain group distribution).
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--use_abs', action='store_true', default=False) # if true, it will use absolute values of satisfied OD as reward (default is to use percentage satsified OD) (does not work in weighted reward)

    args = parser.parse_args()

    environment = Environment(Path(f"./environments/{args.environment}"), groups_file=args.groups_file, group_weights_files=args.group_weights_files, ignore_existing_lines=args.ignore_existing_lines)
    constraints = ForwardConstraints(environment.grid_x_size, environment.grid_y_size, environment.existing_lines_full, environment.grid_to_vector)
    trainer = Trainer(environment, constraints, args)

    # Log parameters on mlflow
    for arg, value in vars(args).items():
        log_param(arg, value)

    if not args.test:
        trainer.train(args)
    else:
        trainer.evaluate(args)

    print("made it!")
