#%%
# To run this file you should be on the root folder of the project and then run python ./environments/xian/prepare_environment.py
# NOTE: run this file AFTER running generate_ams_environment.py

import sys
sys.path.append('./')

# from constants import constants
from pathlib import Path
# import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from environment import Environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

house_price_bins = 5

if __name__ == "__main__":
    # Note: Run this file
    env_path = Path(f"./environments/amsterdam")
    environment = Environment(env_path)

    price_mx = environment.price_mx.cpu().clone().numpy()
    price_mx[price_mx <= 0] = np.nan

    bins = np.quantile(price_mx[~np.isnan(price_mx)], np.linspace(0, 1, house_price_bins + 1))[:-1]
    price_mx_binned = np.digitize(price_mx, bins).astype(np.float32)
    price_mx_binned[np.isnan(price_mx)] = np.nan

    with open(env_path / f'./price_groups_{house_price_bins}.txt', 'w+') as f:
        for i in range(price_mx_binned.shape[0]):
            for j in range(price_mx_binned.shape[1]):
                if not np.isnan(price_mx_binned[i, j]):
                    f.write(f'{i},{j},{price_mx_binned[i,j]}\n')

    # Plot the group membership by square.
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(price_mx_binned)
    
    values = np.unique(price_mx_binned[~np.isnan(price_mx_binned)])
    labels = list(bins)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title("Amsterdam - House price bin groups")
    plt.savefig(env_path / 'house_price_groups.png', bbox_inches='tight')


# %%
