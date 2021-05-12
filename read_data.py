import numpy as np 

dataset = 'data/poland_warszawa_2019_ursynow.pb'

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
        cost_dict[proj] = cost
        
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

cost_dict = create_cost_dict(dataset)
profiles = create_ballots(dataset) 
