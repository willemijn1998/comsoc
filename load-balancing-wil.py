# load balancing 
from utils import *
import numpy as numpy
import random

n = 10
p = 3
costs = [1,2,3]
projects = [project for project in range(p)]
budget = 3
cost_dict = {project:cost for project,cost in zip(projects,costs)}
# Get list of possible ballots 
possible_ballots = feas_set(projects, budget, cost_dict) 
print(possible_ballots) 

profiles = random.choices(possible_ballots, k=n)
print(profiles)



def load_balancing(A, b, proj_costs): 
    """

    A: list of input approval ballots
    b: budget (integer)
    proj_costs: dictionary of projects and costs {project1: cost1, project2: cost2, ...}
    """
    n = len(A)
    

