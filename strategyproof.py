import random
import argparse
from approval_mechanisms import greedy, load_balancing, max_approval
from utils import k_set, feas_set
import numpy as np
from tqdm import tqdm


def get_payoff(profile, prod_costs, budget, manipulator_profile, approval_mechanism, p, n):
    """Get the 'points' the manipulator gets for submitting a profile.

    Input
        :param profile              (list of tuples of ints) the approval ballots per voter, of length n
        :param prod_costs           (dict of int to int) the cost per product
        :param budget               (int) budget, if not provided this is 20% of the total cost
        :param manipulator_profile  (tuple of ints) the true approval ballot for the manipulator
        :param approval_mechanism   (string) the type of approval mechanism to use to select the elected products, must
        :param p                    (int) number of products
        :param n                    (int) number of voters

    Output
        :param payoff               (int) the amount of money spent on projects that are elected that the manipulator voted for

    """

    if approval_mechanism == 'greedy':
        elected = greedy(profile, prod_costs, budget)

    elif approval_mechanism == 'load_balancing':
        elected = load_balancing(A=profile, proj_costs=prod_costs, b=budget, projects=list(prod_costs.keys()))

    elif approval_mechanism == 'max_approval':
        elected = max_approval(P=p, A=profile, b=budget, c=np.asarray(list(prod_costs.values())), n=n)

    else:
        raise ValueError(f'No such approval_mechanism: {approval_mechanism}')

    overlap = set(elected) & set(manipulator_profile)
    payoff = sum([prod_costs[p] for p in overlap])
    return payoff


def check_strategy_proof(profile, prod_costs, possible_ballots, n, budget, p, approval_mechanism):
    """Check strategyproofness for given profile.

    Input
        :param profile              (list of tuples of ints) the approval ballots per voter, of length n
        :param prod_costs           (dict of int to int) the cost per product
        :param possible_ballots     (list of tuples of ints) the possible ballots a voter can submit
        :param n                    (int) number of voters
        :param budget               (int) budget, if not provided this is 20% of the total cost
        :param p                    (int) number of products
        :param approval_mechanism   (string) the type of approval mechanism to use to select the elected products, must
                                                be either greedy, max_approval or load_balancing

    Output
        :param strategy_proof       (boolean) whether or not the profile is strategyproof

    """

    strategy_proof = True
    for manipulator in range(n):

        manipulator_profile = profile[manipulator]
        real_outcome = get_payoff(profile, prod_costs, budget, manipulator_profile, approval_mechanism, p, n)

        for possible_ballot in possible_ballots:

            profile[manipulator] = possible_ballot
            outcome = get_payoff(profile, prod_costs, budget, manipulator_profile, approval_mechanism, p, n)

            if outcome > real_outcome:
                strategy_proof = False
                return strategy_proof

    return strategy_proof


def main(p, n, C_max=2, C=None, b=None, sample_size=100, profile=None, cost_dict=None, approval_mechanism='greedy'):
    """Run the strategyproof test for the provided parameters

    Input
        :param p                    (int) number of products
        :param n                    (int) number of voters
        :param C_max                (int) maximum price a product can be worth, only used to sample uniformly if C is not provided
        :param C                    (list of ints) the costs of all the products, the length must be the same as p
        :param b                    (int) budget, if not provided this is 20% of the total cost
        :param sample_size          (int) sample size to determine strategyproofness percentage
        :param profile              (list of tuples of ints) the approval ballots per voter, of length n
        :param cost_dict            (dict of int to int) the cost per product
        :param approval_mechanism   (string) the type of approval mechanism to use to select the elected products, must
                                                be either greedy, max_approval or load_balancing

    Output
        :param strategyproofness    (int) percentage of the runs (length sample_size) that are strategyproof

    """
    products = list(range(p))
    costs = np.random.randint(1, C_max, size=p) if not C else list(map(int, C))
    budget = b if b else int(0.2 * sum(costs)) + 1  # TODO min cost of a prod must be larger than budget

    cost_dict = cost_dict if cost_dict else dict(zip(products, costs))
    possible_ballots_ = feas_set(products, budget, cost_dict)
    profile_ = profile if profile else random.choices(possible_ballots_, k=n)

    total = []
    for _ in tqdm(range(sample_size)):
        total.append(
            check_strategy_proof(
                n=n,
                budget=b,
                profile=profile_,
                prod_costs=cost_dict,
                possible_ballots=possible_ballots_,
                approval_mechanism=approval_mechanism
            )
        )

    return sum(total) / len(total) * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--p', type=int, help='number of product', default=3)
    parser.add_argument('--C', type=list, help='cost per product', default=None)
    parser.add_argument('--C_max', type=int, help='max cost of a product', default=2)
    parser.add_argument('--b', type=int, help='budget', default=None)
    parser.add_argument('--n', type=int, help='number of voters', default=3)
    parser.add_argument('--sample_size', type=int, help='number of checks on strategyproofness', default=1000)
    parser.add_argument('--approval_mechanism', type=str, help='type of approval mechanism', default='greedy')

    args = parser.parse_args()

    percentage = main(p=args.p, n=args.n, C_max=args.C_max, C=args.C, b=args.b, approval_mechanism=args.approval_mechanism)
    print(f'The provided situation is strategyproof: {percentage}%')
