#%%
import itertools
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
ams_nb = gpd.read_file('./ams-districts.geojson', crs='EPSG:4326')
# ams_nb = gpd.read_file('./ams-neighbourhoods.geojson')
# For population/income data by buurt/wijk
ams_ses = pd.read_csv('./ams-ds-ses.csv')
# ams_ses['p_dutch_w_migr'] = ams_ses['p_dutch'] + ams_ses['p_w_migr']
ams_ses['nr_dutch_w'] = ams_ses['nr_dutch'] + ams_ses['nr_w_migr']
ams_ses = ams_ses[['WK_CODE', 'pop', 'avg_inc_per_res', 'house_price', 'nr_dutch_w', 'nr_nw_migr']]


# Amersfoort / RD New -- Netherlands - Holland - Dutch
CRS = 'EPSG:28992'
# Flatten the neighborhoods
ams_nb = ams_nb.to_crs(CRS)
# %%
xmin, ymin, xmax, ymax = ams_nb.total_bounds

# n = 45
# cell_size = (xmax-xmin)/n
cell_size = 500

cols = list(np.arange(xmin, xmax + cell_size, cell_size))
rows = list(np.arange(ymin, ymax + cell_size, cell_size))

grid = {'geometry': [], 'lat': [], 'lon': [], 'g_x': [], 'g_y': [], 'v': []}
# total cell counter/index
v = 0 
# We reverse the rows here because we want the index of the cells to start from top to bottom.
for i, y in enumerate(reversed(rows[:-1])):
    for j, x in enumerate(cols[:-1]):
        grid['lat'].append(y)
        grid['lon'].append(x)
        grid['g_x'].append(i)
        grid['g_y'].append(j)
        grid['v'].append(v)
        v += 1
        grid['geometry'].append(Polygon([(x,y), (x+cell_size, y), (x+cell_size, y+cell_size), (x, y + cell_size)]))

grid = gpd.GeoDataFrame(grid, crs=CRS)
grid_x_size = grid.g_x.max() + 1
grid_y_size = grid.g_y.max() + 1

metro_stops = pd.read_csv('./TRAMMETRO_PUNTEN_2022.csv', delimiter=';')
metro_stops = metro_stops[metro_stops['Modaliteit'] == 'Metro']
metro_stops = gpd.GeoDataFrame(
                metro_stops, 
                geometry=gpd.points_from_xy(metro_stops['LNG'], 
                metro_stops['LAT']),
                crs='EPSG:4326')
metro_stops = metro_stops.to_crs(CRS)

fig, ax = plt.subplots(figsize=(15, 10))
ams_nb.plot(ax=ax)
grid.boundary.plot(ax=ax, edgecolor='gray')
metro_stops.plot(ax=ax, color='orange', markersize=40)
fig.suptitle(f'Amsterdam Environment Grid - Total Cells: {grid.shape[0]}', fontsize=25)
fig.tight_layout()
# ax.set_axis_off()
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_wijk.png')

# Create metro lines
metro_labels = ['50', '51', '52', '53', '54']
# Sequence of stations in each metro line. Not provided in the dataset so we hard-code it using the OBJECTNUMMER column.
# List of lists where each list corresponds to a line in metro_labels accordingly. 
# E.g. metro_sequence[0] -> the sequence of stations for line metro_labels[0]
metro_sequences = [
        [162, 166, 168, 167, 171, 161, 170, 169, 172, 152, 173, 164, 191, 178, 174, 175, 176, 177, 165, 163], #50
        [162, 166, 168, 167, 171, 161, 170, 169, 172, 152, 173, 164, 181, 190, 185, 184, 183, 189, 182], #51
        [219, 218, 182, 220, 221, 222, 217, 152], #52
        [182, 189, 183, 184, 185, 190, 181, 191, 196, 213, 192, 193, 194, 195], #53
        [182, 189, 183, 184, 185, 190, 181, 191, 178, 174, 175, 176, 177, 165, 163], #54
    ]
# Convert each list of OBJECTNUMMER to a dictionary of {OBJECTNUMMER:0, ...} to merge with main dataframe
metro_sequences = [{k: v for v, k in enumerate(ms)} for ms in metro_sequences]
metro_sequences = [pd.DataFrame(ms.items(), columns=['OBJECTNUMMER', 'Seq']) for ms in metro_sequences]

metro_lines = []
for i, label in enumerate(metro_labels):
    l = metro_stops[metro_stops['Lijn_select'].str.contains(label)]
    l = l.merge(metro_sequences[i]).sort_values('Seq')
    metro_lines.append(l)

#%% Overlay the grid over the neighborhoods to calculate population by grid
# https://gis.stackexchange.com/questions/421888/getting-the-percentage-of-how-much-areas-intersects-with-another-using-geopandas

grid['area_grid'] = grid.area
if CRS == 'EPSG:28992':
    grid['area_grid_km'] = grid['area_grid'] / 10**6
ams_nb['area_nb'] = ams_nb.area

overlay = grid.overlay(ams_nb, how='intersection')
overlay['area_joined'] = overlay.area
overlay['area_overlay_pct'] = overlay['area_joined'] / overlay['area_nb']

overlay_pct = (overlay
           .groupby(['v','WK_CODE'])
           .agg({'area_overlay_pct':'sum'}))

# Plot the distribution of covered neighborhoods per grid
counts, edges, bars = plt.hist(
    overlay_pct.value_counts('v').values, 
    weights=np.ones(len(overlay_pct.value_counts('v').values)) / len(overlay_pct.value_counts('v').values))
counts = counts.round(2)
plt.bar_label(bars, labels=counts)
plt.title('Distribution of covered neighborhoods per square grid')
plt.savefig('./grid_to_nb_distribution.png')

# Assign population to each grid cell.
overlay_pct = overlay_pct.reset_index().merge(ams_ses, on='WK_CODE', how='left')
overlay_pct['grid_pop'] = overlay_pct['area_overlay_pct'] * overlay_pct['pop']
overlay_pct['grid_pop'] = overlay_pct['grid_pop'].round()
gridpop = overlay_pct.groupby('v')[['grid_pop']].sum().reset_index()

# Assign population of nw, w-dutch people to each grid cell.
overlay_pct['nr_dutch_w'] = (overlay_pct['area_overlay_pct'] * overlay_pct['nr_dutch_w']).round()
dutchwpop = overlay_pct.groupby('v')[['nr_dutch_w']].sum().reset_index()
dutchwpop['p_dutch_w'] = dutchwpop['nr_dutch_w']/dutchwpop['nr_dutch_w'].sum()

overlay_pct['nr_nw'] = (overlay_pct['area_overlay_pct'] * overlay_pct['nr_nw_migr']).round()
nwpop = overlay_pct.groupby('v')[['nr_nw']].sum().reset_index()
nwpop['p_nw'] = nwpop['nr_nw']/nwpop['nr_nw'].sum()

#%%
# Assign average income to each grid.
# https://stackoverflow.com/questions/31521027/groupby-weighted-average-and-sum-in-pandas-dataframe
def weighted_average(df, data_col, weight_col, by_col):
    df['_data_times_weight'] = df[data_col] * df[weight_col]
    df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result

gridinc = weighted_average(overlay_pct, 'avg_inc_per_res', 'area_overlay_pct', 'v').to_frame('grid_avg_inc')
gridhouseprice = weighted_average(overlay_pct, 'house_price', 'area_overlay_pct', 'v').to_frame('grid_house_price')

# Merge gridpop with the original grid
grid = grid.merge(gridpop, on='v', how='left')
grid.loc[pd.isna(grid['grid_pop']), 'grid_pop'] = 0
grid = grid.merge(gridinc, on='v', how='left')
grid = grid.merge(gridhouseprice, on='v', how='left')
grid = grid.merge(dutchwpop, on='v', how='left')
grid = grid.merge(nwpop, on='v', how='left')

# Population Density per square
grid['pop_density_km'] = grid['grid_pop'] / grid['area_grid_km']
# grid['grid_pop'] = grid['grid_pop'].fillna(0)
grid.to_csv('./amsterdam_grid.csv', index=False)

#%% Plot the Grid environment and the existing lines.
gridenv = np.zeros((grid_x_size, grid_y_size))
gridenvinc = np.zeros((grid_x_size, grid_y_size))
gridenvhp = np.zeros((grid_x_size, grid_y_size))
gridenvdw = np.zeros((grid_x_size, grid_y_size))
gridevnnw = np.zeros((grid_x_size, grid_y_size))

for i, row in grid.iterrows():
    gridenv[row['g_x'], row['g_y']] = row['grid_pop']
    gridenvinc[row['g_x'], row['g_y']] = row['grid_avg_inc']
    gridenvhp[row['g_x'], row['g_y']] = row['grid_house_price']
    gridenvdw[row['g_x'], row['g_y']] = row['p_dutch_w']
    gridevnnw[row['g_x'], row['g_y']] = row['p_nw']

#%%
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(gridenv, cmap='Blues')
markers = itertools.cycle(['o','s','v', '^', 'P', 'h'])
for i, l in enumerate(metro_lines):
    l_v = l.sjoin(grid)
    l_v = l_v.sort_values('v')

    ax.plot(l_v['g_y'], l_v['g_x'], 'o', marker=next(markers), label=metro_labels[i], markersize=10, alpha=0.5)

fig.suptitle('Amsterdam Grid Population - Existing Lines', fontsize=30)
fig.colorbar(im, orientation='vertical')
ax.legend()

fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_population.png')

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(gridenvinc, cmap='Blues')
markers = itertools.cycle(['o','s','v', '^', 'P', 'h'])
for i, l in enumerate(metro_lines):
    l_v = l.sjoin(grid)
    l_v = l_v.sort_values('v')

    ax.plot(l_v['g_y'], l_v['g_x'], 'o', marker=next(markers), label=metro_labels[i], markersize=10, alpha=0.5)

fig.suptitle('Amsterdam Grid Avg Income - Existing Lines', fontsize=30)
fig.colorbar(im, orientation='vertical')
ax.legend()
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_avg_income.png')

# Save existing lines' grid coordinates to a file.
metro_lines_g = []
for i, l in enumerate(metro_lines):
    l_v = l.sjoin(grid)
    l_g = [list(i) for i in zip(l_v['g_x'].tolist(), l_v['g_y'].tolist())]
    metro_lines_g.append(l_g)

# The contents of this file are then copied to environments/amsterdam/config.txt on existing_lines field
with open('./existing_lines.txt', 'w+') as f:
    for line in metro_lines_g:
        f.write(f"{line}\n")

# Generate and save existing lines full grid coordinates.
# TODO: rework this script - it's very raw and was written in an age of haste, famine and scientific integrity collapse.
# Basically we want to fill the cells between two consecutive metro stops. This helps later on when we look for potential connections to the old lines from the newly created ones.
# e.g if the existing lines have stations in cells [1, 1] and [1, 3], this script will also append [1, 2] in between the other stations.
metro_lines_full = []
for l_i, l in enumerate(metro_lines_g):
    l_full = []

    for i in range(1, len(l)):
        l_full.append(l[i-1])

        xmin = min(l[i-1][0], l[i][0])
        xmax = max(l[i-1][0], l[i][0])

        ymin = min(l[i-1][1], l[i][1])
        ymax = max(l[i-1][1], l[i][1])

        x = range(xmin, xmax)[1:]
        y = range(ymin, ymax)[1:]

        if len(x) == 0 and len(y) == 0:
            continue

        max_len = max(len(x), len(y))
        
        if len(x) == 0:
            x = [xmin]
        if len(y) == 0:
            y = [ymin]
        
        x = list(x)
        y = list(y)

        x.extend([x[0]] * abs(len(x)-len(y)))
        y.extend([y[0]] * abs(len(x)-len(y)))

        for k in range(len(x)):
            l_full.append([x[k], y[k]])
        l_full.append(l[i])
    metro_lines_full.append(l_full)

# The contents of this file are then copied to environments/amsterdam/config.txt on existing_lines_full field
with open('./existing_lines_full.txt', 'w+') as f:
    for line in metro_lines_full:
        f.write(f"{line}\n")

#%%
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(gridenvhp, cmap='Blues')
fig.suptitle('Amsterdam Grid Avg House Price', fontsize=30)
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_avg_house_price.png')


fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(gridenvdw, cmap='Blues')
fig.suptitle('Amsterdam Grid Dutch/Western Population Distribution', fontsize=30)
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_dutch_western_distr.png')

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(gridevnnw, cmap='Blues')
fig.suptitle('Amsterdam Grid Non-Western Population Distribution', fontsize=30)
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_non_western_distr.png')

# %% Print labels of the Grid
fig, ax = plt.subplots(figsize=(50, 40))
# fig2, ax2 = plt.subplots(figsize=(25, 20))
grid['coords'] = grid['geometry'].apply(lambda x: x.representative_point().coords[:])
grid['coords'] = [coords[0] for coords in grid['coords']]

grid.boundary.plot(ax=ax, edgecolor='gray')
for idx, row in grid.iterrows():
    ax.annotate(text=f"{row['v']}\n{row['g_x']},{row['g_y']}", xy=row['coords'],
                 horizontalalignment='center', verticalalignment='center', fontsize=16)

fig.savefig(f'./amsterdam_env{len(rows)}x{len(cols)}_grid_labels.png')
# %% Calculate OD Flows using the Universal Mobility Law
d = 7
fmin = 1/d
fmax = d
od_mx = np.zeros((grid.shape[0], grid.shape[0]))
for i,row_i in grid.iterrows():
    for j, row_j in grid.iterrows():
        if i == j:
            continue
        
        # if row_i['grid_pop'] == 0:
        #     continue

        # destination attractiveness
        mu_j = row_j['pop_density_km'] * row_j['area_grid_km'] ** 2 * fmax
        
        if np.isnan(mu_j):
            mu_j = 0
        
        # Manhattan distance
        r_ij = abs(row_i['g_x'] - row_j['g_x']) + abs(row_i['g_y'] - row_j['g_y'])
        if np.isnan(r_ij):
            print(f'Distance between {i} and {j} is nan - this is a bug and should not happen')
        # Origin Destination flow estimate
        od_ij = mu_j * row_i['area_grid_km'] / r_ij ** 2 * np.log(fmax/fmin)
        od_mx[i, j] = od_ij

# The above code calculates avg visits/day
od_mx = od_mx * d

# Aggregate all flows between 2 locations in a symmetrical od matrix, where od[i,j] = od[j,i]
# This is done because we are designing metro lines which are undirected.
# And also because that's how they do it in the City Metro Network Design paper.
od_mx_sym = np.zeros_like(od_mx)
for i in range(od_mx.shape[0]):
    for j in range(od_mx.shape[1]):
        od_mx_sym[i, j] = od_mx[i, j] + od_mx[j, i]
od_mx_sym = od_mx_sym / 2

# Save OD matrices to file.
with open('./od_bi.txt', 'w') as f:
    for i in range(od_mx.shape[0]):
        for j in range(od_mx.shape[1]):
            if od_mx[i, j] != 0:
                f.write(f'{i},{j},{od_mx[i,j]}\n')

with open('./od.txt', 'w') as f:
    for i in range(od_mx_sym.shape[0]):
        for j in range(od_mx_sym.shape[1]):
            if od_mx_sym[i, j] != 0:
                f.write(f'{i},{j},{od_mx_sym[i,j]}\n')
# %% Plot aggregate OD flow for each grid cell
agg_out_od_g = np.zeros((grid_x_size, grid_y_size))
agg_in_od_g = np.zeros((grid_x_size, grid_y_size))
agg_out_od_v = od_mx.sum(axis=1)
agg_in_od_v = od_mx.sum(axis=0)
# Get the grid indices.
for i in range(agg_out_od_v.shape[0]):
    agg_out_od_g[grid.iloc[i]['g_x'], grid.iloc[i]['g_y']] = agg_out_od_v[i]
    agg_in_od_g[grid.iloc[i]['g_x'], grid.iloc[i]['g_y']] = agg_in_od_v[i]

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(agg_out_od_g, cmap='Blues')
fig.suptitle('Amsterdam Agregate Outgoing OD flows', fontsize=30)
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_agg_out_od_flows.png')

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(agg_in_od_g, cmap='Blues')
fig.suptitle('Amsterdam Agregate Incoming OD flows', fontsize=30)
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_agg_in_od_flows.png')

# %% Plot correlation between population and aggregate d
fig, ax = plt.subplots(figsize=(7, 5))
grid['aggregate_out_od'] = agg_out_od_v
grid['aggregate_in_od'] = agg_in_od_v
grid['aggregate_od'] = grid['aggregate_out_od'] + grid['aggregate_in_od']
corr = grid[['grid_pop', 'aggregate_od']].corr().iloc[0, 1]
grid.plot.scatter('grid_pop', 'aggregate_od', ax=ax)
fig.suptitle(f'Aggregate OD flows vs Population: Pearson: {round(corr, 3)}')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_agg_od_vs_pop.png')

# %% Experiment - Can we correlate aggregate OD with total Checkins/Checkouts?
gvb = pd.read_csv('./gvb_data/check_ins_outs_2019.csv')

gvb.loc[gvb['stop_name'] == 'Atat?rk', 'stop_name'] = 'Atatürk'
gvb.loc[gvb['stop_name'] == 'Burg. R?ellstraat', 'stop_name'] = 'Burg. Röellstraat'
gvb.loc[gvb['stop_name'] == 'Lumi?restraat', 'stop_name'] = 'Lumierestraat'
gvb.loc[gvb['stop_name'] == 'VU medisch centrum', 'stop_name'] = 'VUmc'
gvb.loc[gvb['stop_name'] == 'Fred. Hendrikplants.', 'stop_name'] = 'F. Hendrikplantsoen'
gvb.loc[gvb['stop_name'] == 'Pr. Irenestraat', 'stop_name'] = 'Prinses Irenestraat'
gvb.loc[gvb['stop_name'] == 'Middenhoven / Brink', 'stop_name'] = 'Brink'
gvb.loc[gvb['stop_name'] == 'Nw.Willemsstraat', 'stop_name'] = 'Nw. Willemsstraat'
gvb.loc[gvb['stop_name'] == 'Anth. Moddermanstraat', 'stop_name'] = 'Ant. Moddermanstraat'
gvb.loc[gvb['stop_name'] == "Haarl'meerstation", 'stop_name'] = 'Haarlemmermeerstation'
gvb.loc[gvb['stop_name'] == 'Diemen (Sniep)', 'stop_name'] = 'Diemen Sniep'
gvb.loc[gvb['stop_name'] == 'Ferd. Bolstraat', 'stop_name'] = 'Ferdinand Bolstraat'
gvb.loc[gvb['stop_name'] == 'V. M. Broeckmanstraat', 'stop_name'] = 'V. M. Broekmanstraat'

# total checkins/outs per stop
gvb = gvb.groupby('stop_name')[['check_ins', 'check_outs']].sum().reset_index()
gvb = gvb[gvb['stop_name'] != '[[ Onbekend ]]']
gvb = gvb[gvb['stop_name'] != 'OVan rig']
gvb.loc[:, 'stop_name_lower'] = gvb['stop_name'].str.lower()
gvb.loc[:, 'check_ins_outs'] = gvb['check_ins'] + gvb['check_outs']

pt_stops = pd.read_csv('./ams_network_stops_osm.csv', delimiter=',')

pt_stops = pt_stops[pt_stops['unique_agency_id'] == 'gvb']
pt_stops = pt_stops.groupby('stop_name').mean().reset_index()
pt_stops = pt_stops[['stop_name', 'x', 'y']]
pt_stops[['gemeente', 'stop_name']] = pt_stops['stop_name'].str.split(', ', 1, expand=True)
pt_stops.loc[np.isin(pt_stops['stop_name'],
        ['Osdorpplein Noord', 'Osdorpplein Oost', 'Osdorpplein West']), 'stop_name'] = 'Osdorpplein'
pt_stops.loc[pt_stops['stop_name'] == 'Gelderlandplein O', 'stop_name'] = 'Gelderlandplein Oost'
pt_stops.loc[:, 'stop_name_lower'] = pt_stops['stop_name'].str.lower()


gvb_geo = gvb.merge(pt_stops, on='stop_name_lower').drop(['stop_name_lower', 'stop_name_y'], axis=1)
gvb_geo = gvb_geo.rename(columns={'stop_name_x': 'stop_name'})
gvb_geo = gpd.GeoDataFrame(
            gvb_geo, 
            geometry=gpd.points_from_xy(gvb_geo['x'], gvb_geo['y']),
            crs='EPSG:4326')
gvb_geo = gvb_geo.to_crs(CRS)
gvb_geo = gvb_geo.sjoin(grid)

gvb_stats = gvb_geo.groupby(['v', 'g_x', 'g_y'])[['aggregate_out_od', 'aggregate_in_od', 'aggregate_od', 'check_ins', 'check_outs', 'check_ins_outs']].sum().reset_index()

# Exclude centraal cell - outlier
# TODO SOS: don't have a hard-coded cell here
gvb_stats = gvb_stats[gvb_stats['v'] != 587]
# Exclude station zuid
gvb_stats = gvb_stats[gvb_stats['v'] != 1006]
# Exclude Amstelstation
gvb_stats = gvb_stats[gvb_stats['v'] != 776]

gvb_stats['check_ins_outs_norm'] = gvb_stats['check_ins_outs'] / gvb_stats['check_ins_outs'].max()
gvb_stats['aggregate_out_od_norm'] = gvb_stats['aggregate_out_od'] / gvb_stats['aggregate_out_od'].max()
gvb_stats['aggregate_in_od_norm'] = gvb_stats['aggregate_in_od'] / gvb_stats['aggregate_in_od'].max()
gvb_stats['aggregate_od_norm'] = gvb_stats['aggregate_od'] / gvb_stats['aggregate_od'].max()
gvb_stats['check_ins_norm'] = gvb_stats['check_ins'] / gvb_stats['check_ins'].max()
gvb_stats['check_outs_norm'] = gvb_stats['check_outs'] / gvb_stats['check_outs'].max()

fig, ax = plt.subplots(figsize=(10, 5))
gvb_stats['check_ins_outs_norm'].hist(ax=ax, alpha=0.5, label='Checkins + Checkouts (normalized)')
gvb_stats['aggregate_od_norm'].hist(ax=ax, alpha=0.5, label='Estimated OD Flows (normalized)')
ax.legend(loc='best')
ax.grid(False)
fig.suptitle(f'Distribution of OV Checkins + Checkouts & Estimated OD Flows')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_flow_distribution.png')

#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
corr = gvb_stats[['check_ins_outs_norm', 'aggregate_out_od_norm']].corr().iloc[0, 1]
gvb_stats.plot.scatter('check_outs_norm', 'aggregate_out_od_norm', ax=axs[0], title=f'pearson={gvb_stats[["check_outs_norm", "aggregate_out_od_norm"]].corr().iloc[0, 1].round(2)}')
gvb_stats.plot.scatter('check_ins_norm', 'aggregate_in_od_norm', ax=axs[1], title=f'pearson={gvb_stats[["check_ins_norm", "aggregate_in_od_norm"]].corr().iloc[0, 1].round(2)}')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_checkins_vs_od_flows.png')

fig, ax = plt.subplots(figsize=(10, 4))
gvb_stats.plot.scatter('check_ins_outs_norm', 'aggregate_od_norm', ax=ax, title=f'pearson={gvb_stats[["check_ins_outs_norm", "aggregate_od_norm"]].corr().iloc[0, 1].round(2)}')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_total_checkins_vs_od_flows.png')

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
g_checkins_checkouts = np.zeros((grid_x_size, grid_y_size))
g_checkins = np.zeros((grid_x_size, grid_y_size))
g_checkouts = np.zeros((grid_x_size, grid_y_size))
g_estimated_ods = np.zeros((grid_x_size, grid_y_size))
for i, row in gvb_stats.iterrows():
    g_checkins[int(row['g_x']), int(row['g_y'])] = row['check_ins']
    g_checkouts[int(row['g_x']), int(row['g_y'])] = row['check_outs']
    g_checkins_checkouts[int(row['g_x']), int(row['g_y'])] = row['check_ins_outs_norm']
    g_estimated_ods[int(row['g_x']), int(row['g_y'])] = row['aggregate_out_od_norm'] + row['aggregate_in_od_norm']

im0 = axs[0].imshow(g_checkins_checkouts, label='asd')
axs[0].set_title('GVB | Checkins + Checkouts (normalized)')
axs[1].imshow(g_estimated_ods)
axs[1].set_title('Estimation | Aggregate Flow (normalized)')
cax = fig.add_axes([0.25, 0.25, 0.5, 0.02])
cbar = fig.colorbar(im0, cax=cax, orientation='horizontal', label=f'Note: only contains the {round(gvb_geo.shape[0]/gvb.shape[0] * 100, 1)}% cells where a match was found between GVB and OSM stops datasets')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_gvb_vs_estimate.png')

#%%
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(g_checkins, cmap='Blues')
fig.suptitle('GVB | Checkins', fontsize=30)
ax.set_title('Excluding Centraal, Zuid and Amstelstation')
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_gvb_checkins.png')

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(g_checkouts, cmap='Blues')
fig.suptitle('GVB | Checkouts', fontsize=30)
ax.set_title('Excluding Centraal, Zuid and Amstelstation')
fig.colorbar(im, orientation='vertical')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_gvb_checkouts.png')

# %% Important files for creating the Environment Object
# Average house price per grid cell.
with open('./average_house_price_gid.txt', 'w') as f:
    for i, row in grid.iterrows():
        if np.isnan(row['grid_house_price']):
            continue
        
        f.write(f'{row["g_x"]},{row["g_y"]},{row["grid_house_price"]}\n')

# Population distribution per grid cell.
with open('./dutch_western_popd_gid.txt', 'w') as f:
    for i, row in grid.iterrows():
        if np.isnan(row['p_dutch_w']):
            continue
        
        f.write(f'{row["g_x"]},{row["g_y"]},{row["p_dutch_w"]}\n')

with open('./nonwestern_popd_gid.txt', 'w') as f:
    for i, row in grid.iterrows():
        if np.isnan(row['p_nw']):
            continue
        
        f.write(f'{row["g_x"]},{row["g_y"]},{row["p_nw"]}\n')
