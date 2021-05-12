import collections
from utils import create_appr_dict
import numpy as np
from collections import defaultdict
np.random.seed(4)


def greedy(profile, prod_costs, budget):
    """Greedy approval mechanism that takes a profile (list of approval ballots per voter), dictionary that maps the
    products (ints) to their costs (ints) and a budget (ints). Returns the list of elected products.


    Input
        profile: list of tuples of ints                                             [(int,int), (int), (int, int, int)]
        prod_costs: dictionary that maps each product (int) to their cost (int)     {0:1, 1:3, 2:5}
        budget: int                                                                 10

    Output
        elected: list of ints                                                       [1,2]
    """

    flat_votes = sorted([p for ballot in profile for p in ballot])  # make sure we have lexographical tie breaking

    total_elected = []
    total_cost_elected = 0

    while True:

        if flat_votes == [] or total_cost_elected >= budget:
            break

        prod_elected = collections.Counter(flat_votes).most_common(1)[0][0]
        prod_elected_cost = prod_costs[prod_elected]

        if prod_elected_cost + total_cost_elected <= budget:
            total_elected.append(prod_elected)
            total_cost_elected = total_cost_elected + prod_elected_cost

        flat_votes = list(filter(prod_elected.__ne__, flat_votes))

    return total_elected


def max_approval(P, A, b, c, n):
    '''
    P (int): number of projects
    A (list of array): profile of approval ballots
    b (int): budget
    c (array of int): cost of each project
    n (int): number of voters
    '''
    budget = np.full((P, n * P), np.inf)
    projects = defaultdict(list)
    
    approval_score = np.zeros(P)
    
    for A_i in A:
        approval_score[np.asarray(A_i)] += 1
    
    for k in range(P):
        for t in range(n * P):
            
            if k == 0 and t in approval_score:
                indices = np.where(approval_score == t)
                min_index = np.argmin(c[indices])
                budget[k, t] = c[indices][min_index]
                projects[(k, t)] = [indices[0][min_index]]
                
            elif k > 0:
                min_cost = np.inf
                min_projects_k = []
                
                for p_k in range(P):
                    old_t = int(t - approval_score[p_k])
                    old_projects = projects[(k-1, old_t)]
                    
                    if (old_t) >= 0 and (old_t) < (n * P) and p_k not in old_projects:
                        
                        cost_k = budget[k-1, old_t] + c[p_k]

                        if cost_k < min_cost:
                            min_cost = cost_k
                            copy_k = old_projects.copy()
                            copy_k.append(p_k)
                            min_projects_k = copy_k

                budget[k, t] = np.minimum(budget[k-1, t], min_cost)

                if budget[k, t] == min_cost:
                    projects[(k, t)] = min_projects_k
                else:
                    projects[(k, t)] = projects[(k-1, t)]


    feasable_set = np.where(budget <= b)
    index = np.argmax(feasable_set[1])
    outcome = projects[(feasable_set[0][index], feasable_set[1][index])]
    
    return outcome


def load_balancing(A, b, proj_costs, projects): 
    """
    A: list of tuples with input approval ballots
    b: budget (integer)
    projects: list of projects
    proj_costs: dictionary of projects and costs {project1: cost1, project2: cost2, ...}
    """
    n = len(A)
    A = list(map(list, A))       
    
    # dictionary with {project1: [list of voters that approved], ...}
    appr_dict = create_appr_dict(A, projects) 

    S = [] # accepted projects
    loads = np.zeros(n) # initial loads 
        
    while projects: 
        min_score = 1000000
        new_project = None

        for project in projects: 
            
            score = (proj_costs[project] + np.sum(loads[appr_dict[project]]))/ len(appr_dict[project])
            
            if score < min_score: 
                new_project = project 
                min_score = score 

        if not new_project: 
            break 
                
        if proj_costs[new_project]<= b: 
            S.append(new_project)
            b = b - proj_costs[new_project]
            loads[appr_dict[new_project]] = min_score
            
        projects.remove(new_project)
        
    return S   