import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import argparse


def plot_it(prods=10, voters=[5], cost_iterations=1, C_max=2, sample_size=100, experiment='uniform'):

    df = pd.DataFrame(columns=['n_prods', 'n_voters', 'max_cost', 'type', 'sample_size', '%'])
    now = str(datetime.datetime.now())

    constant = 'prods' if prods == 10 else 'voters'
    not_constant = 'voters' if constant == 'prods' else 'prods'

    for _ in range(cost_iterations):  # add some runs for the cost to be random

        n_products = prods

        for n_voters in tqdm(voters):
            percentage_greedy = main(n_projects=n_products, n_voters=n_voters, C_max=C_max, sample_size=sample_size, approval_mechanism='greedy')
            percentage_load_balancing = main(n_projects=n_products, n_voters=n_voters, C_max=C_max, sample_size=sample_size, approval_mechanism='load_balancing')

            try:
                percentage_max_approval = main(n_projects=n_products, n_voters=n_voters, C_max=C_max, sample_size=sample_size, approval_mechanism='max_approval')
            except:
                percentage_max_approval = None

            df.loc[len(df)] = [n_products, int(n_voters), C_max, 'greedy', sample_size, percentage_greedy]
            df.loc[len(df)] = [n_products, int(n_voters), C_max, 'load_balancing', sample_size, percentage_load_balancing]
            df.loc[len(df)] = [n_products, int(n_voters), C_max, 'max_approval', sample_size, percentage_max_approval]

            df.to_csv(f"data/results_{experiment}_{now}.csv")

    plt.figure(figsize=[10, 10])
    plt.title(f'Strategy proofness (n {constant} = 5)')
    sns.lineplot(data=df, x=f'n_{not_constant}', y='%', hue='type')
    plt.savefig(f"data/figure_{experiment}_{now}.jpg")
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--cost_max', type=int, default=11)
    parser.add_argument('--sample_size', type=int, default=1500)
    parser.add_argument('--n_products', type=int, default=10)
    parser.add_argument('--experiment', type=str,
                        default='uniform', choices=['distributions','uniform'])
    args = parser.parse_args()

    if args.experiment == 'distributions':
        from strategyproof_adept import main_path as main
    elif args.experiment == 'uniform':
        from strategyproof import main

#    range_voters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    range_voters = [20, 30, 40, 50]
    print(f'Running the experiments for voters {range_voters}.')
    plot_it(prods=args.n_products, voters=range_voters, C_max=args.cost_max, sample_size=args.sample_size, experiment=args.experiment)
