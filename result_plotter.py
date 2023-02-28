#%% This is made as a jupyter notebook, it is made to be ran in the VSCODE interactive mode.
# But it still can be ran as a python file.
from collections import defaultdict
from genericpath import isfile
from pathlib import Path
from tokenize import group
from typing import List
import matplotlib.pyplot as plt
import json
import pandas as pd
from environment import Environment
import numpy as np
plt.rcParams.update({'font.size': 18})
import os
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib import cm
import torch
import numpy as np
import matplotlib.lines as mlines
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Color Palette
cp = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
# cp2 = ['#fd7f6f', '#bd7ebe', 'ffb55a', '#7eb0d5', '#b2e061']

# ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]


#%%

def prepare_metrics_df(models: List):
    # args = defaultdict(list)
    args = pd.DataFrame()
    metrics = defaultdict(list)
    for model_path in models:
        with open(Path('result', model_path, 'result_metrics.json')) as json_file:
            data = json.load(json_file)
            metrics['model'].append(model_path)
            metrics['avg_generated_line'].append(data['avg_generated_line'])
            metrics['mean_sat_group_od'].append(data['mean_sat_group_od'])
            metrics['mean_sat_group_od_pct'].append(data['mean_sat_group_od_pct'])
            metrics['mean_sat_od_by_group'].append(data['mean_sat_od_by_group'])
            metrics['mean_sat_od_by_group_pct'].append(data['mean_sat_od_by_group_pct'])
            metrics['group_gini'].append(data['group_gini'])
            metrics['group_pct_gini'].append(data['group_pct_gini'])
            metrics['group_pct_diff'].append(1-abs(data['mean_sat_od_by_group_pct'][0] - data['mean_sat_od_by_group_pct'][1]))
            if 'mean_distance' in data:
                metrics['mean_distance'].append(data['mean_distance'])
            if 'mean_group_distance' in data:
                metrics['mean_group_distance'].append(data['mean_group_distance'])
        
        argpath = Path('result', model_path, 'args.txt')
        if argpath.is_file():
            with open(argpath) as json_file:
                data = json.load(json_file)
                data['model'] = model_path
                args = pd.concat([args, pd.DataFrame(data, index=[0])])
                # args = args.append(data, ignore_index=True)
                # for k in data.keys():
                    # args[k].append(data[k])
                    # args.append(data[k])
            
    
    df_metrics = pd.DataFrame(metrics)
    # df_args = pd.DataFrame(args)
    if args.shape[0] > 0:
        return pd.merge(args, df_metrics, how='right', on='model')
    else:
        return df_metrics

def print_ci(df, col, model: str, z=1.96):
    m = df[col].mean()
    std = df[col].std()
    se = std/math.sqrt(df.shape[0])
    print(f"[{model} - {col}] - Mean: {m} +- {z * se} SE: {se}, CI:({m - z * se}, {m + z * se}), Sample Size: {df.shape[0]}")
    return m

def print_stats(df, model:str):
    od = print_ci(df, 'mean_sat_group_od_pct', model)
    gini = print_ci(df, 'group_pct_gini', model)
    lq = print_ci(df, 'lowest_quintile_sat_od_pct', model)
    print('--------')

    return od, gini, lq

def plot_bar(model_names, model_labels, model_colors, model_hatches, 
            metrics_df: pd.DataFrame, env_name: str, figsize=(10, 5), 
            legend_loc='best',
            group_names=('1st quintile', '2nd', '3rd', '4th', '5th')):
    # https://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side-dodged
    # Width of a bar 
    width = 0.3
    ind = np.arange(len(group_names))
    xpos = ind
    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(model_names):
        results = metrics_df.loc[metrics_df['model'] == model].iloc[0]['mean_sat_od_by_group_pct']
        # TODO change that
        results = [r * 100 for r in  results]
        # position of the bar on the x axis
        # xpos = ind if i == 0 else ind + width
        ax.bar(xpos, results , width, label=model_labels[i], color=model_colors[i], hatch=model_hatches[i])
        xpos = xpos + width

    plt.xlabel('House Price Quintiles', fontsize=32)
    plt.ylabel('% of total satisfied flows', fontsize=32)
    
    plt.title(f"Benefits Distribution among Groups - {env_name}", fontsize=32)
    plt.xticks(ind + width * 3 / 2, group_names)

    ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig

def plot_lines(env: Environment, model_names, model_labels: List, model_colors: List, model_markers: List, metrics_df: pd.DataFrame, env_name, lines=None, figsize=(15, 10), legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=figsize)    
    im1 = ax.imshow(env.grid_groups, cm.get_cmap('viridis'), alpha=0.3)
    labels = ['1st quintile', '2nd quintile', '3rd quintile', '4th quintile', '5th quintile']
    values = (np.unique(env.grid_groups[~np.isnan(env.grid_groups)]))
    grid_colors = [ im1.cmap(im1.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=grid_colors[i], label=labels[i] ) for i in range(len(labels)) ]
    
    ax.legend(handles=patches, loc=legend_loc, prop={'size': 14})
    ax.set_title(f'Generated Lines - {env_name}', fontsize=32)
    if lines:
        for i, l in enumerate(lines):
            # Note here we reverse the dimensions because on scatter plots the horizontal axis is the x axis.
            l_v = env.vector_to_grid(l).cpu()
            label = model_labels[i]
            # label = "_no_legend"
            # if i == 0:
                # label = "Generated Metro Lines"

            ax.plot(l_v[1], l_v[0], f"-{model_markers[i]}", color=model_colors[i], label=label, alpha=0.8, markersize=12, linewidth=4)
    else:
        # for i, l in enumerate(metrics_df['avg_generated_line']):
        for i, model in enumerate(model_names):
        # for i, model in enumerate(metrics_df['model']):
            with open(Path('result', model, 'tour_idx_multiple.txt')) as f:
                l = [int(idx) for idx in f.readline().split(',')]
                # l_v = env.vector_to_grid(torch.tensor(l).reshape(-1,1)).cpu()
                l_v = env.vector_to_grid(torch.tensor([l])).T.cpu()

                label = "_no_legend"
                if i == 0:
                    label = "Generated Metro Lines"

            ax.plot(l_v[:, 1], l_v[:, 0], f"-{model_markers[i]}", color=model_colors[i], label=label, alpha=1, markersize=12, linewidth=4)

    fig.tight_layout()
    return fig
# To create a scatterplot with different custom markers.
# From https://github.com/matplotlib/matplotlib/issues/11155
def plot_scatter(x,y,ax=None, markers=None, labels=None, colors=None, env_name=None, figsize=(12, 8)):
    s = np.repeat(800, len(x))
    fig, ax = plt.subplots(figsize=figsize)
    # if not ax: ax = plt.gca()
    sc = ax.scatter(x,y, s=s, c=colors)
    if (markers is not None) and (len(markers)==len(x)):
        paths = []
        for marker in markers:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)

    ax.set_xlabel('% of total satisfied flows', fontsize=32)
    ax.set_ylabel('Gini Index', fontsize=32)
    fig.suptitle(f'Equity vs Efficiency - {env_name}', fontsize=32)
    ax.set_ylim((0,0.8))


    markers = [mlines.Line2D([], [], color=colors[i], marker=markers[i], linestyle='None',
                            markersize=10, label=labels[i]) for i in range(len(s))]
    ax.legend(handles=markers, prop={'size': 24})
    fig.tight_layout()
    return fig

def create_all_plots(env: Environment, metrics_df: pd.DataFrame, metadata: List, bar_plot_models: List, line_plot_models: List, scatter_plot_models: List, scatter_x: List, scatter_y: List, plot_name_prefix: None, env_name=None, figsize=(12,8), group_names=('1st quintile', '2nd', '3rd', '4th', '5th')):
    metadata = pd.DataFrame(metadata, columns=['label', 'model', 'color', 'pattern', 'marker'])
    metadata.index = metadata['label']

    bar_models = metadata.loc[bar_plot_models]
    bar_fig = plot_bar(
        bar_models['model'].tolist(),
        bar_models['label'].tolist(),
        bar_models['color'].tolist(),
        bar_models['pattern'].tolist(),
        metrics_df[metrics_df.index.isin(bar_models['model'].tolist())], 
        env_name=env_name, 
        figsize=figsize,
        group_names=group_names)

    bar_fig.savefig(f'./{plot_name_prefix}_bar.png')

    line_models = metadata.loc[line_plot_models]
    line_fig = plot_lines(
        env, 
        line_models['model'].tolist(),
        line_models['label'].tolist(), 
        line_models['color'].tolist(), 
        line_models['marker'].tolist(),
        metrics_df[metrics_df.index.isin(line_models['model'].tolist())], 
        env_name=env_name, 
        legend_loc="lower left", 
        figsize=figsize)
    line_fig.savefig(f'./{plot_name_prefix}_lines.png')

    scatter_models = metadata.loc[scatter_plot_models]
    scatter_fig = plot_scatter(
        x=scatter_x,
        y=scatter_y,
        markers=scatter_models['marker'].tolist(),
        labels=scatter_models['label'].tolist(),
        colors=scatter_models['color'].tolist(),
        env_name=env_name,
        figsize=figsize) 

    scatter_fig.savefig(f'./{plot_name_prefix}_scatter.png')

# %% AMSTERDAM
amsterdam = Environment(Path(f"./environments/amsterdam/"), groups_file='price_groups_5.txt')
models = []
paths  = [ f.path for f in os.scandir('./result') if f.is_dir() ]
for p in paths:
    if 'amsterdam' in p:
        models.append(p.split('/')[-1])

metrics_ams = prepare_metrics_df(models)
metrics_ams['mean_sat_group_od_pct'] = metrics_ams['mean_sat_group_od_pct'] * 100
metrics_ams['lowest_quintile_sat_od_pct'] = metrics_ams['mean_sat_od_by_group_pct'].str[0]
metrics_ams['lowest_quintile_sat_od_pct'] = metrics_ams['lowest_quintile_sat_od_pct'] * 100
metrics_ams.index = metrics_ams['model']

# %% Amsterdam Income Quintiles averages calculation
ams_empty_ses1 = metrics_ams[ ((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 1)
                            & (metrics_ams['var_lambda'] == 0)]

ams_empty_ses0 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 0)
                            & (metrics_ams['var_lambda'] == 0)]

ams_empty_ggi_2 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'ggi') 
                            & (metrics_ams['ggi_weight'] == 2)]

ams_empty_var_3 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'group') 
                            & (metrics_ams['var_lambda'] == 3)]

ams_empty_rawls = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'rawls')]


ams_full_ses1 = metrics_ams[ ((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 1)
                            & (metrics_ams['var_lambda'] == 0)]

ams_full_ses0 = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 0)
                            & (metrics_ams['var_lambda'] == 0)]

ams_full_var_3 = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'group') 
                            & (metrics_ams['var_lambda'] == 3)]

ams_full_ggi_2 = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'ggi') 
                            & (metrics_ams['ggi_weight'] == 2)]

ams_full_rawls = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'rawls')]


ams_full_ses0_od, ams_full_ses0_gini, ams_full_ses0_lq =  print_stats(ams_full_ses0, 'ams_full_ses_0')
ams_full_ses1_od, ams_full_ses1_gini, ams_full_ses1_lq =  print_stats(ams_full_ses1, 'ams_full_ses_1')
ams_full_var_3_od, ams_full_var_3_gini, ams_full_var_3_lq = print_stats(ams_full_var_3, 'ams_full_var_3')
ams_full_rawls_od, ams_full_rawls_gini, ams_full_rawls_lq = print_stats(ams_full_rawls, 'ams_full_rawls')
ams_full_ggi_2_od, ams_full_ggi_2_gini, ams_full_ggi_2_lq = print_stats(ams_full_ggi_2, 'ams_full_ggi_2')

ams_empty_ses0_od, ams_empty_ses0_gini, ams_empty_ses0_lq =  print_stats(ams_empty_ses0, 'ams_empty_ses_0')
ams_empty_ses1_od, ams_empty_ses1_gini, ams_empty_ses1_lq =  print_stats(ams_empty_ses1, 'ams_empty_ses_1')
ams_empty_var_3_od, ams_empty_var_3_gini, ams_empty_var_3_lq = print_stats(ams_empty_var_3, 'ams_empty_var_3')
ams_empty_rawls_od, ams_empty_rawls_gini, ams_empty_rawls_lq = print_stats(ams_empty_rawls, 'ams_empty_rawls')
ams_empty_ggi_2_od, ams_empty_ggi_2_gini, ams_empty_ggi_2_lq = print_stats(ams_empty_ggi_2, 'ams_empty_ggi_2')

#%% Amsterdam Full environment

ams_full_plot = [
    ['Max. Effic.',   'amsterdam_20220810_09_33_35.507895', cp[0], '', 'o'],
    ['Access. Index',   'amsterdam_20220807_22_43_37.708173', cp[1], '-', 's'],
    ['Var.Reg',         'amsterdam_20220809_00_40_57.169847', cp[2], '+', '^'],
    ['Rawls',           'amsterdam_20220808_11_53_12.554688', cp[3], 'o', 'v'],
    ['GGI',             'amsterdam_20220810_20_23_40.289417', cp[4], '/', 'D'],
]

create_all_plots(amsterdam, metrics_ams, ams_full_plot, 
    bar_plot_models=['Max. Effic.', 'Rawls', 'GGI'],
    line_plot_models=['Max. Effic.', 'Access. Index', 'Rawls', 'GGI', 'Var.Reg'],
    scatter_plot_models=['Max. Effic.', 'Access. Index', 'Var.Reg', 'Rawls', 'GGI'],
    scatter_x=[ams_full_ses0_od, ams_full_ses1_od, ams_full_var_3_od, ams_full_rawls_od, ams_full_ggi_2_od],
    scatter_y=[ams_full_ses0_gini, ams_full_ses1_gini, ams_full_var_3_gini, ams_full_rawls_gini, ams_full_ggi_2_gini],
    plot_name_prefix='ams_full',
    env_name='Amsterdam')


ams_empty_plot = [
    ['Max. Effic.',   'amsterdam_20220705_18_17_31.196986', cp[0], '', 'o'],
    ['Access. Index',   'amsterdam_20220705_18_25_09.654804', cp[1], '-', 's'],
    ['Var.Reg',         'amsterdam_20220706_11_15_16.765435', cp[2], '+', '^'],
    ['Rawls', 'amsterdam_20220810_09_26_45.963603', cp[3], 'o', 'v'],
    ['GGI',             'amsterdam_20220708_11_21_23.191428', cp[4], '/', 'D'],
]

create_all_plots(amsterdam, metrics_ams, ams_empty_plot, 
    bar_plot_models=['Max. Effic.', 'Rawls', 'GGI'],
    line_plot_models=['Max. Effic.', 'Access. Index', 'Rawls', 'GGI', 'Var.Reg'],
    scatter_plot_models=['Max. Effic.', 'Access. Index', 'Var.Reg', 'Rawls', 'GGI'],
    scatter_x=[ams_empty_ses0_od, ams_empty_ses1_od, ams_empty_var_3_od, ams_empty_rawls_od, ams_empty_ggi_2_od],
    scatter_y=[ams_empty_ses0_gini, ams_empty_ses1_gini, ams_empty_var_3_gini, ams_empty_rawls_gini, ams_empty_ggi_2_gini],
    plot_name_prefix='ams_empty',
    env_name='Amsterdam')


#%% Amsterdam Weighted groups Experiment (Western-non-western)
ams_w_empty_ses1 = metrics_ams[ ((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (~pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 1)
                            & (metrics_ams['var_lambda'] == 0)]

ams_w_empty_ses0 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (~pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 0)
                            & (metrics_ams['var_lambda'] == 0)]

ams_w_empty_ggi_2 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (~pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'ggi') 
                            & (metrics_ams['ggi_weight'] == 2)]

ams_w_empty_var_3 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (~pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['reward'] == 'group') 
                            & (metrics_ams['var_lambda'] == 3)]

ams_w_empty_rawls = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (~pd.isnull(metrics_ams['group_weights_files']))
                            & (metrics_ams['actor_lr'] == 15e-4)
                            & (metrics_ams['reward'] == 'rawls')]

ams_w_empty_ses1_od, ams_w_empty_ses1_gini, ams_w_empty_ses1_lq =  print_stats(ams_w_empty_ses1, 'ams_w_empty_ses1')
ams_w_empty_ses0_od, ams_w_empty_ses0_gini, ams_w_empty_ses0_lq =  print_stats(ams_w_empty_ses0, 'ams_w_empty_ses_0')
ams_w_empty_ggi_2_od, ams_w_empty_ggi_2_gini, ams_w_empty_ggi_2_lq = print_stats(ams_w_empty_ggi_2, 'ams_w_empty_ggi_2')
ams_w_empty_var_3_od, ams_w_empty_var_3_gini, ams_w_empty_var_3_lq = print_stats(ams_w_empty_var_3, 'ams_w_empty_var_3')
ams_w_empty_rawls_od, ams_w_empty_rawls_gini, ams_w_empty_rawls_lq = print_stats(ams_w_empty_rawls, 'ams_w_empty_rawls')

ams_w_empty_plot = [
    ['Max. Effic.',     'amsterdam_20221025_18_07_17.271867', cp[0], '', 'o'],
    ['Access. Index',   'amsterdam_20221026_16_31_11.370638', cp[1], '-', 's'],
    ['Var.Reg',         'amsterdam_20221026_18_17_07.310764', cp[2], '+', '^'],
    ['Rawls',           'amsterdam_20221027_09_51_21.915196', cp[3], 'o', 'v'],
    ['GGI',             'amsterdam_20221026_17_28_17.106600', cp[4], '/', 'D'],
]

create_all_plots(amsterdam, metrics_ams, ams_w_empty_plot, 
    bar_plot_models=['Max. Effic.', 'Rawls', 'GGI'],
    line_plot_models=['Max. Effic.', 'Access. Index', 'Rawls', 'GGI', 'Var.Reg'],
    scatter_plot_models=['Max. Effic.', 'Access. Index', 'Var.Reg', 'Rawls', 'GGI'],
    scatter_x=[ams_w_empty_ses0_od, ams_w_empty_ses1_od, ams_w_empty_var_3_od, ams_w_empty_rawls_od, ams_w_empty_ggi_2_od],
    scatter_y=[ams_w_empty_ses0_gini, ams_w_empty_ses1_gini, ams_w_empty_var_3_gini, ams_w_empty_rawls_gini, ams_w_empty_ggi_2_gini],
    plot_name_prefix='ams_w_empty',
    env_name='Amsterdam',
    group_names=('Dutch/Western', 'Non-Western'))


#%% XIAN

xian = Environment(Path(f"./environments/xian/"), groups_file='price_groups_5.txt')
models = []
paths  = [ f.path for f in os.scandir('./result') if f.is_dir() ]
for p in paths:
    if 'xian' in p:
        models.append(p.split('/')[-1])

metrics_xian = prepare_metrics_df(models)
metrics_xian['mean_sat_group_od_pct'] = metrics_xian['mean_sat_group_od_pct'] * 100
metrics_xian['lowest_quintile_sat_od_pct'] = metrics_xian['mean_sat_od_by_group_pct'].str[0]
metrics_ams['lowest_quintile_sat_od_pct'] = metrics_ams['lowest_quintile_sat_od_pct'] * 100
metrics_xian.index = metrics_xian['model']

xian_full_ses0 = metrics_xian[((metrics_xian['existing_lines'] != 0) & ~np.isnan(metrics_xian['existing_lines']))
                            & (metrics_xian['reward'] == 'weighted') 
                            & (metrics_xian['ses_weight'] == 0)
                            & (metrics_xian['var_lambda'] == 0)]

xian_full_ses1 = metrics_xian[((metrics_xian['existing_lines'] != 0) & ~np.isnan(metrics_xian['existing_lines']))
                            & (metrics_xian['reward'] == 'weighted') 
                            & (metrics_xian['ses_weight'] == 1)
                            & (metrics_xian['var_lambda'] == 0)]

xian_full_var_5 = metrics_xian[((metrics_xian['existing_lines'] != 0) & ~np.isnan(metrics_xian['existing_lines']))
                            & (metrics_xian['reward'] == 'group') 
                            & (metrics_xian['var_lambda'] == 5)]

xian_full_ggi_4 = metrics_xian[((metrics_xian['existing_lines'] != 0) & ~np.isnan(metrics_xian['existing_lines']))
                            & (metrics_xian['reward'] == 'ggi') 
                            & (metrics_xian['ggi_weight'] == 4)]

xian_full_rawls = metrics_xian[((metrics_xian['existing_lines'] != 0) & ~np.isnan(metrics_xian['existing_lines']))
                            & (metrics_xian['reward'] == 'rawls')]

xian_full_ses0_od, xian_full_ses0_gini, xian_full_ses0_lq =  print_stats(xian_full_ses0, 'xian_full_ses_0')
xian_full_ses1_od, xian_full_ses1_gini, xian_full_ses1_lq =  print_stats(xian_full_ses1, 'xian_full_ses_1')
xian_full_var_5_od, xian_full_var_5_gini, xian_full_var_5_lq = print_stats(xian_full_var_5, 'xian_full_var_5')
xian_full_rawls_od, xian_full_rawls_gini, xian_full_rawls_lq = print_stats(xian_full_rawls, 'xian_full_rawls')
xian_full_ggi_4_od, xian_full_ggi_4_gini, xian_full_ggi_4_lq = print_stats(xian_full_ggi_4, 'xian_full_ggi_4')



xian_full_plot = [
    ['Max. Effic.',   'xian_20220812_09_42_57.652815', cp[0], '', 'o'],
    ['Access. Index',   'xian_20220812_14_44_22.783845', cp[1], '-', 's'],
    ['Var.Reg',         'xian_20220811_22_41_02.456631', cp[2], '+', '^'],
    ['Rawls', 'xian_20220813_09_28_43.208981', cp[3], 'o', 'v'],
    # ['GGI',             'xian_20220812_18_59_40.094535', cp[4], '/', 'D'],
    ['GGI',             'xian_20220814_12_10_27.594976', cp[4], '/', 'D'],
]

create_all_plots(xian, metrics_xian, xian_full_plot, 
    bar_plot_models=['Max. Effic.', 'Rawls', 'GGI'],
    line_plot_models=['Max. Effic.', 'Access. Index', 'Var.Reg', 'Rawls', 'GGI'],
    scatter_plot_models=['Max. Effic.', 'Access. Index', 'Var.Reg', 'Rawls', 'GGI'],
    scatter_x=[xian_full_ses0_od, xian_full_ses1_od, xian_full_var_5_od, xian_full_rawls_od, xian_full_ggi_4_od],
    scatter_y=[xian_full_ses0_gini, xian_full_ses1_gini, xian_full_var_5_gini, xian_full_rawls_gini, xian_full_ggi_4_gini],
    plot_name_prefix='xian_full', 
    env_name="Xi'an")
#%%
# ams_empty_ggi_2[['actor_lr', 'critic_lr', 'mean_sat_group_od_pct', 'group_gini', 'budget', 'existing_lines', 'ignore_existing_lines']].sort_values('mean_sat_group_od_pct')


# #%%
# # TODO: transfer this method to the environment class.

# dilemma = Environment(Path(f"./environments/dilemma_5x5"))

# def calculate_agg_od(environment):
#     """Calculate aggregate origin-destination flow matrix for each grid square of the given environment.

#     Args:
#         environment (Environment): environment for which to calcualte aggregate OD per grid square.

#     Returns:
#         torch.Tensor: aggregate od by grid
#     """
#     # 
#     # A measure of importance of each square.
#     agg_od_g = torch.zeros((environment.grid_x_size, environment.grid_y_size)).to(device)
#     agg_od_v = environment.od_mx.sum(axis=1)
#     # Get the grid indices.
#     for i in range(agg_od_v.shape[0]):
#         g = environment.vector_to_grid(torch.Tensor([i])).type(torch.int32)
#         agg_od_g[g[0], g[1]] = agg_od_v[i]

#     return agg_od_g

# dilemma_od = calculate_agg_od(dilemma).cpu()
# fig, ax = plt.subplots(figsize=(10, 10))

# im0 = ax.imshow(dilemma_od, cm.get_cmap('Blues'))
# ax.set_xticks(np.arange(-.5, 4, 1))
# ax.set_yticks(np.arange(-.5, 4, 1))
# ax.set_xticklabels(np.arange(0, 5, 1))
# ax.set_yticklabels(np.arange(0, 5, 1))
# ax.grid(color='gray', linewidth=2)
# # cax = fig.add_axes([0.65, 0.175, 0.2, 0.02])
# fig.colorbar(im0, orientation='vertical', fraction=0.046, pad=0.04)
# ax.set_title('(A) Aggregate Origin-Destination Flow')

# # %% PLOT XIAN RESULTS
# xian = Environment(Path(f"./environments/xian/"), groups_file='price_groups_5.txt')
# # metrics_xian = prepare_metrics_df(['old_xian_16_01_47.060823', 'old_xian_16_03_51.724205', 'old_xian_10_48_03.743589', 'old_xian_16_22_50.580720'])
# metrics_xian = prepare_metrics_df(['old_xian_16_22_50.580720', 'old_xian_10_48_03.743589'])
