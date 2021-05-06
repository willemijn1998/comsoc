import random
import argparse
from approval_mechanisms import greedy
from utils import k_set


def check_strategyproofness(profile, prod_costs, possible_ballots, n, budget):

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--p', type=int, help='number of product', default=3)
    parser.add_argument('--C', type=list, help='cost per product', default=[1,1,1])
    parser.add_argument('--b', type=int, help='budget', default=2)
    parser.add_argument('--n', type=int, help='number of voters', default=3)
    parser.add_argument('--k', type=int, help='max number of products on a ballot', default=2)
    parser.add_argument('--sample_size', type=int, help='number of checks on strategyproofness', default=10)

    args = parser.parse_args()

    products = list(map(str, list(range(args.p))))
    costs = list(map(int, args.C))

    cost_dict = dict(zip(products, costs))
    possible_ballots_ = list(k_set(products, args.k))
    profile_ = random.choices(possible_ballots_, k=args.n)


    result = [check_strategyproofness(profile_, cost_dict, possible_ballots_, args.n, args.b) for i in range(args.sample_size)]
    print(f'The provided situation is strategyproof: {sum(result)/len(result) * 100}%')

