# Fairness in Transport Network Design using Deep Reinforcement Learning

### Abstract
Transportation systems fundamentally impact human well-being, economic productivity, and environmental sustainability. Designing well-functioning transport networks is however non-trivial, given the large space of solutions and constraints - ranging from physical to social and legal. Moreover, different spatial segregation sources can render some transportation network interventions unfair to specific groups. It is thereby crucial to optimize the transportation system while mitigating the eventual disproportional benefits that transportation network design can lead to. In this paper, we explore the application of Deep Reinforcement Learning (Deep RL) to the Transport Network Design Problem (TNDP) and the trade-off between utility (satisfied total demand) and fairness (equity in distribution among groups). We formulate the problem as a sequential decision-making task, apply a model based on the actor-critic framework,  and test new (fairness) reward functions inspired by theoretical notions of 1) Utilitarianism, 2) Equal Sharing of Benefits, 3) Narrowing the Gap, and 4) Rawl’s theory of justice. We apply our method to data from two different cities (Amsterdam and Xi'An) and a synthetic environment. We show that vanilla Deep RL leads to biased outcomes and introducing different definitions of fair rewards can lead to different compromises between fairness and utility.


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
