import numpy as np

def get_vec_profiles(n_voters, n_projects, costs, budget, n_profiles, distr=None):
    
    profiles = np.zeros((n_voters, n_profiles, n_projects))

    for i in range(n_voters):
        profiles_i = np.zeros((n_profiles, n_projects))
        
        for j in range(n_profiles):
            c_i = np.inf

            while c_i > budget:
                k_i = np.random.randint(1, n_projects)
                profile_ij = np.random.choice(n_projects, k_i, replace=False, p=distr)
                c_i = costs[profile_ij].sum()
                
            profiles_i[j, profile_ij] = 1

        profiles[i] = profiles_i
        
    return profiles