import numpy as np
np.random.seed(4)

import torch
print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Seed manually to make runs reproducible
# You need to set this again if you do multiple runs of the same model
torch.manual_seed(0)

# When running on the CuDNN backend two further options must be set for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def vec_greedy(P, A, c, b):
    """Greedy approval mechanism that takes a profile (list of approval ballots per voter), dictionary that maps the
    projects (ints) to their costs (ints) and a budget (ints). Returns the list of elected projects.
    Input
        profile: list of tuples of ints                                             [(int,int), (int), (int, int, int)]
        proj_costs: dictionary that maps each project (int) to their cost (int)     {0:1, 1:3, 2:5}
        budget: int                                                                 10
    Output
        elected: list of ints                                                       [1,2]
    """
    with torch.no_grad():
        A = torch.from_numpy(A).to(device).int()
        c = torch.from_numpy(c).to(device).int()

        # Array with approval score: profile x project
        approval_score = torch.sum(A, dim=0).float()

        lex_mask = torch.arange(0, 0.9, 0.9/P).to(device)
        approval_score += lex_mask

        sort_proj = torch.argsort(approval_score, descending=True)
        
        _, n_P, _ = A.shape
        total_cost = torch.zeros((n_P, )).to(device).int()
        projects = torch.zeros((n_P, P)).to(device).int()
        
        for i in range(len(c)):
            i_proj = sort_proj[:, i]
            i_cost = c[i_proj]
            total_cost += i_cost
            
            exceed_indices = torch.where(total_cost > b)[0].long()
            suffice_indices = torch.where(total_cost <= b)[0].long()
            
            
            if len(exceed_indices) != 0:
                total_cost[exceed_indices] -= i_cost[exceed_indices]
                
            if len(suffice_indices) != 0:
                projects[suffice_indices, i_proj[suffice_indices]] = 1
            
    return projects


def vec_max_approval(n, P, A, c, b):
    with torch.no_grad():
        inf = 1000000
        A = torch.from_numpy(A).to(device).int()
        c = torch.from_numpy(c).to(device).int()

        # Array with approval score: profile x project
        approval_score = torch.sum(A, dim=0)
        
        _, n_P, _ = A.shape
        budget = torch.full((n_P, P, n * P), inf).to(device).int()
        projects = torch.zeros((n_P, P, n * P, P)).to(device).int()
        
        for k in range(P):
            for t in range(n * P):
                score = t + 1
                
                if k == 0 and score in approval_score:
                    indices = torch.where(approval_score == score)
                    
                    temp_cost = torch.full((n_P, P), inf).to(device).int()
                    temp_cost[indices] = c[indices[1]]
                    
                    min_index = torch.argmin(temp_cost, dim=1)
                    
                    budget[:, k, t] = temp_cost[torch.arange(n_P), min_index]
                    projects[torch.arange(n_P), k, t, min_index] = 1
                    
                elif k > 0:

                    budget[:, k, t] = budget[:, k-1, t]
                    projects[:, k, t] = projects[:, k-1, t]
                    
                    min_costs = torch.full((n_P, ), inf).to(device).int()
                    min_projects = torch.zeros((n_P, P)).to(device).int()
                    
                    old_t = t - approval_score
                        
                    # To deal with negative values of t
                    non_neg_indices = torch.where(old_t >= 0)

                    if len(non_neg_indices[0]) == 0:
                        continue

                    # To make sure each project is only used once
                    old_projects = projects[non_neg_indices[0], k-1, old_t[non_neg_indices].long(), \
                                            non_neg_indices[1]]
                    old_indices = torch.where(old_projects == 0)[0]

                    if len(old_indices) == 0:
                        continue

                    comb_indices0 = non_neg_indices[0][old_indices]
                    comb_indices1 = non_neg_indices[1][old_indices]

                    costs_k = torch.full((n_P, P), inf).to(device).int()
                    costs_k[comb_indices0, comb_indices1] = budget[comb_indices0, k-1, \
                                                                      old_t[comb_indices0, comb_indices1].long()]
                    costs_k[comb_indices0, comb_indices1] += c[comb_indices1]
                    
                    min_index = torch.argmin(costs_k, axis=1)
                    new_indices = torch.where(costs_k[torch.arange(n_P), min_index] < budget[:, k-1, t])[0]
                    
                    if len(new_indices) != 0:
                        budget[new_indices.long(), k, t] = costs_k[torch.arange(n_P), min_index][new_indices]
                        
                        projects[new_indices.long(), k, t] = projects[new_indices.long(), k-1, \
                                                              old_t[torch.arange(n_P), min_index][new_indices].long()]
                        projects[new_indices.long(), k, t, min_index[new_indices]] = 1
                    
        feasable_set = torch.where(budget <= b, torch.arange(1, n*P+1, 1).to(device).int(), torch.zeros(n*P).to(device).int())
        
        max_index_column = torch.argmax(feasable_set, dim=2)
        max_index_row = torch.argmax(max_index_column, dim=1)
        
        outcome = projects[torch.arange(n_P), max_index_row, max_index_column[torch.arange(n_P), max_index_row]]
    
    return outcome


def vec_load_balancing(n, P, A, c, b): 
    """
    A: list of tuples with input approval ballots
    b: budget (integer)
    projects: list of projects
    proj_costs: dictionary of projects and costs {project1: cost1, project2: cost2, ...}
    """
    with torch.no_grad():
        inf = 1000000.0
        A = torch.from_numpy(A).to(device).int()
        c = torch.from_numpy(c).to(device).float()

        # Array with approval score: profile x project
        approval_score = torch.sum(A, dim=0)
        
        _, n_P, _ = A.shape
    
        accepted_projects = torch.zeros((n_P, P)).to(device).float()
        loads = torch.zeros((n, n_P)).to(device).float()
        
        for _ in range(P):
            
            i_loads = torch.zeros((n, n_P, P)).to(device).float()

            for i in range(P):

                voters_i = torch.where(A[:, :, i] == 1)
                i_loads[voters_i[0], voters_i[1], i] += c[i] + loads[voters_i]
                i_loads[voters_i[0], voters_i[1], i] /= approval_score[voters_i[1], i]

            max_i = torch.argmax(i_loads, dim=0)
            max_i_loads = i_loads[max_i, torch.vstack([torch.arange(n_P)]*P).T, torch.vstack([torch.arange(P)]*n_P)].double()
            
            np_loads = torch.where(accepted_projects == 1, inf, max_i_loads)
            
            min_i = torch.argmin(np_loads, dim=1).long()
            
            accepted_projects[torch.arange(n_P), min_i] = 1
            
            total_c = c @ accepted_projects.T
            
            exceed_indices = torch.where(total_c > b)[0].long()
            suffice_indices = torch.where(total_c <= b)[0].long()
            
            if len(exceed_indices) != 0:
                accepted_projects[torch.arange(n_P)[exceed_indices], min_i[exceed_indices]] = 0
                
            if len(suffice_indices) != 0:
                loads[:, torch.arange(n_P)[suffice_indices]] = i_loads[:, torch.arange(n_P)[suffice_indices], \
                                                                    min_i[suffice_indices]]
            
    return accepted_projects.int()
