import numpy as np 
from utils import *

def create_cost_dict(dataset): 
    """
    Creates cost dictionary from polish data set. Input is name of data set. 
    Output: {project: cost, ...}
    """
    file = open(dataset)
    all_lines = file.readlines()
    
    cost_dict = {}

    start = all_lines.index("PROJECTS\n")+2
    stop = all_lines.index("VOTES\n")
    
    for line in all_lines[start:stop]: 
        proj, cost = line.split(';')[0:2]
        cost_dict[int(proj)] = int(cost)
        
    return cost_dict

def create_ballots(dataset): 
    """
    Creates list of ballots that people voted for. 
    """
    ballots = []
    
    file = open(dataset)
    all_lines = file.readlines()
    
    start = all_lines.index("VOTES\n")+2
    
    for line in all_lines[start:]: 
        ballot = line.split(';')[4].split(',')
        ballot = list(map(int, ballot))
        ballots.append(ballot)
        
    return ballots

# HOW to call these in a function: 

# dataset = 'data/poland_warszawa_2019_ursynow.pb'
# budget = 2000000    
# cost_dict = create_cost_dict(dataset)
# projects = list(cost_dict.keys())
# profiles = create_ballots(dataset) 
# possible_ballots = feas_set(projects, budget, cost_dict)
