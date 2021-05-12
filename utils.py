from itertools import chain, combinations


def k_set(prods, k):
    """Returns the set of possible combinations of products on a ballot with the maximum amount of
    products being k."""
    return chain.from_iterable(combinations(prods, n) for n in range(1, k + 1))\

def feas_set(projects, budget, cost_dict): 
    """
    Returns the set of feasible approval votes of projects on a ballot. The set is bound by the 
    budget, i.e. a voter cannot choose more projects than fits in the available budget. The input dict 
    proj_cost represent the projects and their costs.
    """
    k = 1 + len(projects)
    feas_proj_combs = []

    for proj_comb in chain.from_iterable(combinations(projects, n) for n in range(1, k)):
        cost = sum([cost_dict.get(proj) for proj in proj_comb])
        if cost <= budget: 
            feas_proj_combs.append(proj_comb)

    return feas_proj_combs 