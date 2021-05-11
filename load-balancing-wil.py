# load balancing 
from utils import *
import numpy as numpy

n = 10
p = 3
costs = [1,2,3]
projects = [project for project in range(p)]
budget = 3
cost_dict = {project:cost for project,cost in zip(projects,costs)}
# Get list of possible ballots 
possible_ballots = feas_set(projects, budget, cost_dict) 
print(possible_ballots) 



def load_balancing(R, b, c): 
    n = len(R)