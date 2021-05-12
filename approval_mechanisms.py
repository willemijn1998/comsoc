import collections
from utils import create_appr_dict


def greedy(profile, prod_costs, budget):

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
    """ 
    Dit is de klad versie, snap het nu eindelijk
    Zal het morgen beter implementeren
    """
    budget = np.full((P, n * P), np.inf)
    projects = collections.defaultdict(list)
    
    approval_score = np.zeros(P)
    
    for A_i in A:
        approval_score[A_i] += 1
    
    for k in range(P):
        for t in range(n * P):
            
            if k == 0 and t in approval_score:
                indices = np.where(approval_score == t)
                min_cost = np.min(c[indices])
                budget[k, t] = min_cost
                
                projects[(k,t)] = [[proj] for proj in indices[0]]
                
            elif k > 0:
                min_budget_i = np.inf
                min_projects_k = []
                
                for p_k in range(P):
                    new_t = int(t - approval_score[p_k])
                    budget_i = np.inf
                    projects_k = []
                    
                    if (new_t) >= 0 and (new_t) < (n * P):
                        for proj_k in projects[(k-1, new_t)]:
                            
                            if p_k not in proj_k:
                                budget_i = budget[k-1, new_t] + c[k]
                                copy_k = proj_k.copy()
                                copy_k.append(p_k)
                                projects_k.append(copy_k)
                                
                    if budget_i < min_budget_i:
                        min_budget_i = budget_i
                        min_projects_k = projects_k

                budget[k, t] = np.minimum(budget[k-1, t], min_budget_i)

                if budget[k-1, t] <= min_budget_i:
                    projects[(k, t)] = projects[(k-1, t)]
                    
                else:
                    projects[(k, t)] = min_projects_k
                
    indices = np.where(budget <= b)
    index = np.argmax(indices[1])
    return projects[(indices[0][index], indices[1][index])]


def load_balancing(A, b, proj_costs, projects): 
    """
    A: list of tuples with input approval ballots
    b: budget (integer)
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

        for project in projects: 
            
            score = (proj_costs[project] + np.sum(loads[appr_dict[project]]))/ len(appr_dict[project])
            
            if score < min_score: 
                # ensures lexicographic tiebreaking 
                new_project = project 
                min_score = score 
                
        if proj_costs[new_project]<= b: 
            S.append(new_project)
            b = b - proj_costs[new_project]
            loads[appr_dict[new_project]] = min_score
            
        projects.remove(new_project)
        
    return S   