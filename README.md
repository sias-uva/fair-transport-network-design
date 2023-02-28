# Fairness in Transport Network Design using Deep Reinforcement Learning

### Abstract
Optimizing urban transportation networks can improve the lives of millions of citizens worldwide. Designing well-functioning transport networks is however non-trivial, given the large space of solutions and constraints. Moreover, different spatial segregation sources can render transportation network interventions unfair to specific groups. It is thereby crucial to mitigate the disproportional benefits that transportation network design can lead to. In this paper, we explore the application of fairness in the Transport Network Design Problem (TNDP) through Reinforcement Learning (RL). The combination of RL with the formalization of different fairness definitions as reward functions --- inspired by 1) Equal Sharing of Benefits, 2) Narrowing the Gap, and 3) Rawl’s justice theory --- allows us to explore the trade-offs between efficiency (satisfied total demand) and fairness (distribution among groups) for a wide range of scenarios. We apply our method to data from two different cities (Amsterdam and Xi'An) and a synthetic environment. We show that \textit{vanilla} Deep RL can lead to biased outcomes and considering different definitions of fair rewards results in different compromises between fairness and efficiency.

![](https://github.com/sias-uva/fair-transport-network-design/blob/main/result_images/ams_xian.png)

### Setup
There are three files: `environment-mac.txt`, `environment-windows.txt`, `environment-cross-platform.yml`, which can be used to initialize the conda environment and run the scripts. For Linux, the cross-platform file should create the desired environment.

The Xi'an and Amsterdam city environments are already preprocessed and prepared.

### Training 
Here are some examples to run the training process for different environments/reward functions:

`python main.py --environment=xian --groups_file=price_groups_5.txt --budget=210 --reward=weighted --ses_weight=0 --station_num_lim=45 --epoch_max=3500`

`python main.py --environment=amsterdam --groups_file=price_groups_5.txt --budget=125 --reward=weighted --ses_weight=1 --station_num_lim=20 --epoch_max=3500`

`python main.py --environment=amsterdam --groups_file=price_groups_5.txt --budget=125 --reward=ggi --ggi_weight=4 --station_num_lim=20 --epoch_max=3500`

### Testing
`python main.py --environment=xian --result_path=xian_20220814_12_31_19.406095 --test --groups_file=price_groups_5.txt --budget=210 --station_num_lim=45`

Where `result_path` should be replaced with the path of the trained model (automatically created on the result folder).

### Replicating reported results
The reported results were obtained using the following argument setup:
|           | Reward Function | actor_lr | critic_lr | reward   | ses_weight | var_lambda | ggi_weight |
|-----------|-----------------|----------|-----------|----------|-----------:|------------|------------|
| Xi'an     | Maximize OD     | 10e-4    | 10e-4     | weighted |          0 |          - |          - |
|           | Maximize Equity | 10e-4    | 10e-4     | weighted |          1 |          - |          - |
|           | Var. Reg.       | 10e-4    | 10e-4     | group    |          - |          5 |          - |
|           | Lowest Quintile | 10e-4    | 10e-4     | rawls    |          - |          - |          - |
|           | GGI             | 10e-4    | 10e-4     | ggi      |          - |          - |          4 |
| Amsterdam | Maximize OD     | 10e-4    | 10e-4     | weighed  |          0 |          - |          - |
|           | Maximize Equity | 10e-4    | 10e-4     | weighted |          1 |          - |          - |
|           | Var. Reg.       | 10e-4    | 10e-4     | group    |          - |          3 |          - |
|           | Lowest Quintile | 15e-4    | 15e-4     | rawls    |          - |          - |          - |
|           | GGI             | 10e-4    | 10e-4     | ggi      |          - |          - |          2 |
