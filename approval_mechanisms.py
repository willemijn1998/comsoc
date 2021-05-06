import collections


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
