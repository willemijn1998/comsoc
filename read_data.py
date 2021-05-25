import numpy as np 
import random as rd

def read_data(path): 
    file = open(path)
    all_lines = file.readlines()
    return all_lines

def get_costs(dataset): 
    """
    Creates cost dictionary from polish data set. Input is name of data set. 
    Output: {project: cost, ...}
    """    
    cost_dict = {}

    start = dataset.index("PROJECTS\n")+2
    stop = dataset.index("VOTES\n")
    
    costs = []
    projects = []
    
    for line in dataset[start:stop]: 
        projects.append(int(line.split(';')[0]))
        costs.append(int(line.split(';')[1]))
        
    return costs, projects

def get_ballots(dataset): 
    """
    Creates list of ballots that people voted for. 
    """
    ballots = []
    
    start = dataset.index("VOTES\n")+2
    
    for line in dataset[start:]: 
        ballot = line.split(';')[4].split(',')
        ballot = list(map(int, ballot))
        ballots.append(ballot)
        
    return ballots

def get_votes(dataset): 
    """
    Read the amount of votes per project. 
    """
    
    projects = []
    votes = []

    start = dataset.index("PROJECTS\n")+2
    stop = dataset.index("VOTES\n")
    
    for line in dataset[start:stop]: 
        projects.append(int(line.split(';')[0]))
        votes.append(int(line.split(';')[3]))
        
    return votes, projects

def get_budget(dataset): 
    budget = int(dataset[10][7:-1])
    return budget
    
def exhaustiveness(ballots, cost_dict, budget): 
    
    exhaust = []
    
    for ballot in ballots: 
        total_costs = 0
        for project in ballot: 
            total_costs += cost_dict[project]
        exhaust.append(total_costs/budget)
        
    return exhaust

def get_distributions(path, n_projects): 
    dataset = read_data(path)
    ballots = get_ballots(dataset)
    budget = get_budget(dataset)
    votes, projects = get_votes(dataset)
    costs, projects = get_costs(dataset)
    cost_dict = {proj:cost for proj,cost in zip(projects,costs)}
    vote_dict = {proj:vote for proj,vote in zip(projects,votes)}
    budget_percentage = budget/sum(costs)
    
    # Exhaustiveness 
    exhaust = exhaustiveness(ballots, cost_dict, budget)
    exh_distr, bins, _ = plt.hist(exhaust, bins=10)
    exh_values = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    
    # Popularity 
    projects_sample = rd.choices(projects, k=n_projects)
    costs_sample = []
    votes_sample = []
    for project in projects_sample: 
        costs_sample.append(cost_dict[project])
        votes_sample.append(vote_dict[project])
        
    sorted_pairs = sorted(zip(votes_sample, costs_sample), reverse=True)
    pop_distr, costs = [list(tupl) for tupl in zip(*sorted_pairs)]   
    
    return exh_distr, exh_values, pop_distr, costs, budget_percentage


def create_synth_profile(path, n_voters, n_projects): 
    """ 
    This functiion creates profiles from a pabulib file 
    """
    exh_distr, exh_values, pop_distr, costs, budget_percentage = get_distributions(path, n_projects) 
    projects = [i for i in range(n_projects)]
    
    budget = sum(costs) * budget_percentage
    profile = []

    for i in range(n_voters): 
        exh = rd.choices(exh_values, exh_distr)[0]
        max_cost = exh * budget
        total_cost = 0 
        ballot = []
        
        while True: 
            project = rd.choices(projects, pop_distr, k=1)[0]
            ballot.append(project)
            total_cost += costs[project]
            
            if total_cost >= max_cost: 
                break
            
        profile.append(ballot)
    
    return profile, costs, budget

