# Base case 
import numpy 
import random as rd
from itertools import chain


# amount of voters
n = 10
# amount of projects 
p = 3
# list of projects
P = [p for p in range(p)]
# costs 
C = [1, 2, 3]
# create dictionary with costs 
c = {p:c for p,c in zip(P,C)}
print(c)
# budget
b = 3


def create_R(n, costs, budget): 
    """
    Create a set of rankings for unit cost case. It creates rankings assuming that people approve as many as feasible. 
    """
    R = []
    for voter in range(n): # loop through all voters
        r = []
        c = dict(costs)
        b = budget
        while c:
            # select random alternative from list of alternatives
            x = rd.choice(list(c.keys()))
            r.append(x)
            # subtract cost from budget 
            b = b - c[x]
            # remove approved project from list 
            c.pop(x) 

            # Create new dictionary without projects that exceed the budgets
            c = {project:cost for project, cost in c.items() if cost <= b}

        R.append(r)

    return R 

R = create_R(n, c, b)
print("These are the profiles:", R)
c = {p:c for p,c in zip(P,C)}

 
def greedy_approval(R, b, c): 
    projects = list(chain.from_iterable(R))
    projects = sorted(projects)
    S = []
    while projects: 
        x = max(projects, key=projects.count)
        if c[x] <= b: 
            S.append(x)
            b = b - c[x]
        projects = [p for p in projects if p!= x]

    return S

S = greedy_approval(R, b, c)
print("These are chosen:", S)
