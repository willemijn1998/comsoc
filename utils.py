import numpy as np
from itertools import chain, combinations, compress


def k_set(prods, k):
    """Returns the set of possible combinations of products on a ballot with the maximum amount of
    products being k."""
    return chain.from_iterable(combinations(prods, n) for n in range(1, k + 1))


def feas_set(projects, budget, cost_dict): 
    """Returns the set of feasible approval votes of projects on a ballot. The set is bound by the
    budget, i.e. a voter cannot choose more projects than fits in the available budget. The input dict 
    proj_cost represent the projects and their costs."""
    options = list(chain.from_iterable(combinations(projects, n) for n in range(1, len(projects) + 1)))
    ballot_costs = np.fromiter(map(lambda x: sum(list(map(cost_dict.get, x))), options), dtype=np.int)
    return set(compress(options, ballot_costs <= budget))


def create_appr_dict(A, projects): 
    """Creates a dictionary of projects and the voters that approved of them: {project1: [list of voters], ...}
    A: list of tuples with input approval ballots, first in list corresponds to first voter
    projects: list of projects"""
    n = len(A)
    appr_dict = {p:[] for p in projects}
    
    for voter,ballot in enumerate(A): 
        for proj in ballot: 
            appr_dict[proj].append(voter)
    
    return appr_dict 
