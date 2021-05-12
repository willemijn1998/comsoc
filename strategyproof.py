import random
import argparse
from approval_mechanisms import greedy
from utils import k_set
import numpy as np


def check_strategy_proof(profile, prod_costs, possible_ballots, n, budget):

    strategy_proof = True
    for manipulator in range(n):

        manipulator_profile = profile[manipulator]

        real_elected = greedy(profile, prod_costs, budget)
        real_overlap = set(real_elected) & set(manipulator_profile)
        real_outcome = sum([prod_costs[p] for p in real_overlap])

        for possible_ballot in possible_ballots:

            profile[manipulator] = possible_ballot
            elected = greedy(profile, prod_costs, budget)
            overlap = set(elected) & set(manipulator_profile)
            outcome = sum([prod_costs[p] for p in overlap])

            if outcome > real_outcome:
                strategy_proof = False
                return strategy_proof

    return strategy_proof


def main(p, n, C_max=2, C=None, k=None, b=None, sample_size=100, profile=None, cost_dict=None):

    products = list(map(str, list(range(p))))
    costs = np.random.randint(1, C_max, size=p) if not C else list(map(int, C))
    k = k if k else p
    b = b if b else int(0.2 * sum(costs)) + 1

    cost_dict = cost_dict if cost_dict else dict(zip(products, costs))
    possible_ballots_ = list(k_set(products, k))
    profile_ = profile if profile else random.choices(possible_ballots_, k=n)

    return [check_strategy_proof(profile_, cost_dict, possible_ballots_, n, b) for _ in range(sample_size)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--p', type=int, help='number of product', default=3)
    parser.add_argument('--C', type=list, help='cost per product', default=None)
    parser.add_argument('--C_max', type=int, help='max cost of a product', default=2)
    parser.add_argument('--b', type=int, help='budget', default=None)
    parser.add_argument('--n', type=int, help='number of voters', default=3)
    parser.add_argument('--k', type=int, help='max number of products on a ballot', default=None)
    parser.add_argument('--sample_size', type=int, help='number of checks on strategyproofness', default=1000)

    args = parser.parse_args()

    result = main(args.p, args.n, args.C_max, args.C, args.k, args.b)
    print(f'The provided situation is strategyproof: {sum(result) / len(result) * 100}%')
