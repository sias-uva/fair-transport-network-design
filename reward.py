from os import environ
from re import L
from environment import Environment
import torch
from utils import gini_tensor, gini

from constants import device

def od_utility(tour_idx: torch.Tensor, environment: Environment):
    """Total sum of satisfied Origin Destination flows.

    Args:
        tour_idx (torch.Tensor): the generated line
        environment (Environment): the environment where the line is generated.

    Returns:
        torch.Tensor: sum of satisfied Origin Destination Flows.
    """
    sat_od_mask = environment.satisfied_od_mask(tour_idx)
    reward = (environment.od_mx * sat_od_mask).sum().to(device)
    
    return reward

def group_utility(tour_idx: torch.Tensor, environment: Environment, var_lambda=0, use_pct=True, mult_gini=False):
    """Sums total satisfied Origin Destination flows of all groups 
    (equal to od_utility in cases where every square with a group also has OD flows), 
    and subtracts a lambda of the variance (to achieve minimization of differences fairness.)

    Args:
        tour_idx (torch.Tensor): the generated line.
        environment (Environment): the environment where the line is generated.
        var_lambda (int, optional): variance weight parameter to subtract from the sum. Defaults to 0.
        use_pct (boolean, optional): if True, reward will be calculated using percentage of satisfied OD per group. If false, it will use absolute values. Defaults to True.
        mult_gini (boolean, optional): if True, it will multiply the group utility by 1-gini_index(group utility), as they do on the AI economist paper.

    Returns:
        torch.Tensor: total reward.
    """
    assert environment.group_od_mx, 'Cannot use group_utility reward without group definitions. Provide --groups_file argument'

    sat_od_mask = environment.satisfied_od_mask(tour_idx)

    sat_group_ods = torch.zeros(len(environment.group_od_mx), device=device)
    sat_group_ods_pct = torch.zeros(len(environment.group_od_mx), device=device)
    for i, g_od in enumerate(environment.group_od_mx):
        sat_group_ods[i] = (g_od * sat_od_mask).sum().item()
        sat_group_ods_pct[i] = sat_group_ods[i] / g_od.sum()

    if use_pct:
        group_rw = sat_group_ods_pct
    else:
        group_rw = sat_group_ods

    if mult_gini:
        rw = group_rw.sum() * (1 - gini(group_rw.detach().cpu().numpy()))
        if torch.isnan(rw):
            return 0
        return rw
    else:
        return group_rw.sum() - var_lambda * group_rw.var()

def lowest_quintile_utility(tour_idx: torch.Tensor, environment: Environment, use_pct=True, group_idx=0):
    """Based on Rawl's theory of justice - returns the satisfied OD (or % depending on use_pct) of the lowest quintile.

    Args:
        tour_idx (torch.Tensor): the generated line.
        environment (Environment): the environment where the line is generated.
        use_pct (boolean, optional): if True, reward will be calculated using percentage of satisfied OD per group. If false, it will use absolute values. Defaults to True.
        group_idx (int, optional): Which group to optimize for -- defaults to the first one.
    Returns:
        torch.Tensor: total reward.
    """
    assert environment.group_od_mx, 'Cannot use group_utility reward without group definitions. Provide --groups_file argument'

    sat_od_mask = environment.satisfied_od_mask(tour_idx)

    sat_od = (environment.group_od_mx[group_idx] * sat_od_mask).sum().item()
    if use_pct:
        return sat_od / environment.group_od_mx[group_idx].sum()
    else:
        return sat_od


def discounted_development_utility(tour_idx: torch.Tensor, environment: Environment, p=2.0):
    """Total sum of utility as defined by City Metro Network Expansion with Reinforcement Learning paper.
   For each covered square in the generated line, calculate the distance to the other covered squares,
   discount it and multiply it with the average house price of each square.

    Args:
        tour_idx (torch.Tensor): the generated line.
        environment (Environment): the environment where the line is generated.
        p (float, optional): p-norm distance to calculate: 1: manhattan, 2: euclidean, etc. Defaults to 2.0.

    Returns:
        torch.Tensor: sum of total discounted development utility
    """
    tour_idx_g = environment.vector_to_grid(tour_idx).transpose(0, 1)
    tour_ses = environment.price_mx_norm[tour_idx_g[:, 0], tour_idx_g[:, 1]]

    # total_util = torch.zeros(tour_idx_g.shape[0], device=device)
    total_util = torch.zeros(1, device=device)
    for i in range(tour_idx_g.shape[0]):
        # Calculate the distance from each origin square to every other square covered by the line.
        distance = torch.cdist(tour_idx_g[i][None, :].float(), tour_idx_g.float(), p=p).squeeze()
        # Discount squares based on their distance. (-0.5 is theoretically a tunable parameter)
        discount = torch.exp(-0.5 * distance)
        discount[i] = 0 # origin node should have no weight in the final calculation of the utility.

        # total_util[i] = (discount * tour_ses).sum()
        total_util += (discount * tour_ses).sum()

    return total_util

def ggi(tour_idx: torch.Tensor, environment: Environment, weight, use_pct=True):
    """Generalized Gini Index reward (see paper for more information).
    Exponentially smaller weights are assigned to the groups with the highest satisfied origin-destination flows.

    Args:
        tour_idx (torch.Tensor): the generated line.
        environment (Environment): the environment where the line is generated: 1/weight^index in order
        weight (int): weight base to use on the calculation of GGI: 
        use_pct (bool, optional): if True, reward will be calculated using percentage of satisfied OD per group. If false, it will use absolute values. Defaults to True.

    Returns:
        torch.Tensor: total ggi
    """
    sat_od_mask = environment.satisfied_od_mask(tour_idx)

    sat_group_ods = torch.zeros(len(environment.group_od_mx), device=device)
    sat_group_ods_pct = torch.zeros(len(environment.group_od_mx), device=device)
    for i, g_od in enumerate(environment.group_od_mx):
        sat_group_ods[i] = (g_od * sat_od_mask).sum().item()
        sat_group_ods_pct[i] = sat_group_ods[i] / g_od.sum()

    if use_pct:
        group_rw = sat_group_ods_pct
    else:
        group_rw = sat_group_ods

    # Generate weights for each group.
    weights = torch.tensor([1/(weight**i) for i in range(group_rw.shape[0])], device=device)
    # "Normalize" weights to sum to 1
    weights = weights/weights.sum()
    
    group_rw, _ = torch.sort(group_rw)
    reward = torch.sum(group_rw * weights)

    if use_pct:
        reward *= 1000

    return reward

# def group_weighted_utility(tour_idx: torch.Tensor, environment: Environment, var_lambda=0, use_pct=True, mult_gini=False):
#     """

#     Args:
#         tour_idx (torch.Tensor): the generated line.
#         environment (Environment): the environment where the line is generated.
#         var_lambda (int, optional): variance weight parameter to subtract from the sum. Defaults to 0.
#         use_pct (boolean, optional): if True, reward will be calculated using percentage of satisfied OD per group. If false, it will use absolute values. Defaults to True.
#         mult_gini (boolean, optional): if True, it will multiply the group utility by 1-gini_index(group utility), as they do on the AI economist paper.

#     Returns:
#         torch.Tensor: total reward.
#     """
#     assert environment.group_weights, 'Cannot use group_weighted_utility reward without group weights. Provide --group_weights_files argument'

#     sat_od_mask = environment.satisfied_od_mask(tour_idx)

#     sat_group_ods = torch.zeros(len(environment.group_od_mx), device=device)
#     sat_group_ods_pct = torch.zeros(len(environment.group_od_mx), device=device)
#     for i, g_od in enumerate(environment.group_od_mx):
#         sat_group_ods[i] = (g_od * sat_od_mask).sum().item()
#         sat_group_ods_pct[i] = sat_group_ods[i] / g_od.sum()

#     if use_pct:
#         group_rw = sat_group_ods_pct
#     else:
#         group_rw = sat_group_ods

#     if mult_gini:
#         rw = group_rw.sum() * (1 - gini(group_rw.detach().cpu().numpy()))
#         if torch.isnan(rw):
#             return 0
#         return rw
#     else:
#         return group_rw.sum() - var_lambda * group_rw.var()