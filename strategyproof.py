from utils import get_vec_profiles
import argparse
import itertools
from approval_mechanisms import vec_greedy, vec_load_balancing, vec_max_approval
from utils import *
import numpy as np
from tqdm import tqdm


def get_payoff(n, p, profile, costs, budget, manipulator_ballot, approval_mechanism):
    """Get the 'points' the manipulator gets for submitting a profile.
    Input
        :param profile              (list of tuples of ints) the approval ballots per voter, of length n
        :param proj_costs           (dict of int to int) the cost per project
        :param budget               (int) budget, if not provided this is 20% of the total cost
        :param manipulator_profile  (tuple of ints) the true approval ballot for the manipulator
        :param approval_mechanism   (string) the type of approval mechanism to use to select the elected projects, must
        :param p                    (int) number of projects
        :param n                    (int) number of voters
    Output
        :param payoff               (int) the amount of money spent on projects that the manipulator voted for
    """

    if approval_mechanism == 'greedy':
        elected = vec_greedy(p, profile, costs, budget)

    elif approval_mechanism == 'load_balancing':
        elected = vec_load_balancing(n, p, profile, costs, budget)

    elif approval_mechanism == 'max_approval':
        elected = vec_max_approval(n=n, P=p, A=profile, c=costs, b=budget)

    else:
        raise ValueError(f'No such approval_mechanism: {approval_mechanism}')

    elected = elected.cpu().numpy()
    elected = np.repeat(np.expand_dims(elected, axis=1), len(manipulator_ballot), axis=1)
    overlap_mask = np.where(elected == manipulator_ballot, np.arange(1, p+1), 0)
    overlap = elected * overlap_mask

    zero_costs = np.hstack([0, costs])
    payoff = zero_costs[overlap]
    payoff = payoff.sum(axis=2)
    
    return payoff


def check_strategy_proof(n, p, profile, costs, budget, approval_mechanism):
    """Check strategy proofness for given profile.
    Input
        :param profile              (list of tuples of ints) the approval ballots per voter, of length n
        :param proj_costs           (dict of int to int) the cost per project
        :param possible_ballots     (list of tuples of ints) the possible ballots a voter can submit
        :param n                    (int) number of voters
        :param budget               (int) budget, if not provided this is 20% of the total cost
        :param p                    (int) number of projects
        :param approval_mechanism   (string) the type of approval mechanism to use to select the elected projects, must
                                                be either greedy, max_approval or load_balancing
    Output
        :param strategy_proof       (boolean) whether or not the profile is strategy proof
    """
    strategy_proof = True
    for manipulator in range(n):

        manipulator_ballot = profile[manipulator]
        mb_indices = np.where(manipulator_ballot == 1)[1]

        feasible_ballots = []
        for ballot in np.array(list(itertools.product([0, 1], repeat=p))):
            if sum(costs[np.where(ballot == 1)[0]]) <= budget:
                feasible_ballots.append(ballot)

        manipulation_ballots = []
        for ballot in feasible_ballots:

            _1 = set(np.where(ballot == 1)[0]).issubset(mb_indices)
            _2 = np.sum(ballot) != 0
            _3 = set(mb_indices).issubset(np.where(ballot == 1)[0])
            _4 = np.sum(ballot) == np.sum(manipulator_ballot) + 1
            if (_1 and _2) or (_3 and np.sum(ballot) == _4):
                manipulation_ballots.append(ballot)

        real_outcome = get_payoff(
            n, p, profile=profile, costs=costs, budget=budget,
            manipulator_ballot=manipulator_ballot, approval_mechanism=approval_mechanism
        ).diagonal()
        
        start = 0
        stop = 0

        for stop in np.arange(100, len(manipulation_ballots), 100):
            profiles = np.repeat(profile, 100, axis=1)
            profiles[manipulator] = np.vstack(np.repeat([manipulation_ballots[start: stop]], profile.shape[1], axis=0))
            outcome = get_payoff(n, p, profiles, costs, budget, manipulator_ballot, approval_mechanism)
            succes_indices = np.where(outcome > real_outcome)

            if len(succes_indices[0]) != 0:
                strategy_proof = False
                return strategy_proof

            start = stop

        profiles = np.repeat(profile, len(manipulation_ballots)-stop, axis=1)
        profiles[manipulator] = np.vstack(np.repeat([manipulation_ballots[stop:]], profile.shape[1], axis=0))
        outcome = get_payoff(n, p, profiles, costs, budget, manipulator_ballot, approval_mechanism)
        succes_indices = np.where(outcome > real_outcome)

        if len(succes_indices[0]) != 0:
            strategy_proof = False
            return strategy_proof

    return strategy_proof


def main(n_voters, n_projects, C_max=2, sample_size=100,
         approval_mechanism='greedy', distr=None, C=None, b=None):
    """Run the strategy proof test for the provided parameters
    Input
        :param n_projects         (int) number of projects
        :param n_voters           (int) number of voters
        :param C_max              (int) maximum price for a projects, only used to sample uniformly if C is not provided
        :param C                  (list of ints) the costs of all the projects, the length must be the same as p
        :param b                  (int) budget, if not provided this is 20% of the total cost
        :param sample_size        (int) sample size to determine strategyproofness percentage
        :param profile            (list of tuples of ints) the approval ballots per voter, of length n
        :param cost_dict          (dict of int to int) the cost per project
        :param approval_mechanism (string) the type of approval mechanism to use to select the elected projects, must
                                                be either greedy, max_approval or load_balancing
    Output
        :param strategy_proofness (int) percentage of the runs (length sample_size) that are strategy proof
    """
    costs = np.random.randint(1, C_max, size=n_projects) if not C else np.array(C).astype(int)

    min_budget = int(2 * min(costs))  # min budget must be at least twice the price of the cheapest project
    suggested_budget = int(0.3 * sum(costs))  # budget should be 30% of the total costs based on real world data
    budget = b if b else max(suggested_budget, min_budget)

    total = []

    print(f'Running {approval_mechanism}:')

    for _ in tqdm(range(sample_size)):
        profiles = get_vec_profiles(n_voters, n_projects, costs, budget, 1, distr)

        total.append(
            check_strategy_proof(
                n=n_voters,
                p=n_projects,
                profile=profiles,
                costs=costs,
                budget=budget,
                approval_mechanism=approval_mechanism
            )
        )

    return sum(total) / len(total) * 100


def main_path(n_voters, n_projects, path='./input_data/poland_warszawa_2018_ursynow-wysoki-polnocny.pb',
              approval_mechanism='greedy', sample_size=100):
    """Run the strategy proof test for the provided parameters
    Input
        :param n_projects         (int) number of projects
        :param n_voters           (int) number of voters
        :param sample_size        (int) sample size to determine strategyproofness percentage
        :param profile            (list of tuples of ints) the approval ballots per voter, of length n
        :param cost_dict          (dict of int to int) the cost per project
        :param approval_mechanism (string) the type of approval mechanism to use to select the elected projects, must
                                                be either greedy, max_approval or load_balancing
    Output
        :param strategy_proofness (int) percentage of the runs (length sample_size) that are strategy proof
    """
    dataset = read_data(path)
    total = []

    print(f'Running {approval_mechanism}:')

    for _ in tqdm(range(sample_size)):
        profiles_, costs, budget_ = create_synth_profile_vec(dataset, n_voters, n_projects)
        min_budget = int(max(costs))
        budget = max(min_budget, budget_)

        total.append(
            check_strategy_proof(
                n=n_voters,
                p=n_projects,
                profile=profiles_,
                costs=costs,
                budget=budget,
                approval_mechanism=approval_mechanism
            )
        )

    return sum(total) / len(total) * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--n_voters', type=int, help='number of voters', default=3)
    parser.add_argument('--n_projects', type=int, help='number of projects', default=3)
    parser.add_argument('--cost_max', type=int, help='max cost of a project', default=2)
    parser.add_argument('--sample_size', type=int, help='number of checks on strategy proofness', default=1000)
    parser.add_argument('--approval_mechanism', type=str, help='type of approval mechanism', default='greedy')
    parser.add_argument('--distr', type=list, help='popularity distribution per project', default=None)
    parser.add_argument('--costs', type=list, help='cost per project', default=None)
    parser.add_argument('--budget', type=int, help='budget', default=None)
    parser.add_argument('--path', type=str, help='Dataset to import for distribution', default=None)

    args = parser.parse_args()

    if args.path:
        percentage = main_path(
            n_voters=args.n_voters,
            n_projects=args.n_projects,
            path=args.path,
            approval_mechanism=args.approval_mechanism,
            sample_size=args.sample_size
        )
    else:
        percentage = main(
            n_projects=args.n_projects,
            n_voters=args.n_voters,
            C_max=args.cost_max,
            C=args.C,
            b=args.b,
            approval_mechanism=args.approval_mechanism,
            sample_size=args.sample_size
        )

    print(f'The provided situation is strategy proof: {percentage}%')
