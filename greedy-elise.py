from itertools import chain, combinations
import random
import collections
import argparse


def k_set(products, k):
    """Returns the set of possible combinations of products on a ballot with the maximum amount of
    products being k."""
    return chain.from_iterable(combinations(products, n) for n in range(1, k + 1))


def greedy_approval(p, n, b, C, k):

    products = list(map(str, list(range(p))))
    costs = list(map(int, C))
    assert len(products) == len(costs)

    cost_dict = dict(zip(products, costs))
    possible_sets = list(k_set(products, k))
    profiles = random.choices(possible_sets, k=n)
    flat_votes = sorted([p for ballot in profiles for p in ballot])

    total_elected = []
    total_cost_elected = 0
    stop = False

    while not stop:

        if flat_votes == [] or total_cost_elected >= b:
            return total_elected, total_cost_elected

        prod_elected = collections.Counter(flat_votes).most_common(1)[0][0]
        prod_elected_cost = cost_dict[prod_elected]

        if prod_elected_cost + total_cost_elected <= b:
            total_elected.append(prod_elected)
            total_cost_elected = total_cost_elected + prod_elected_cost

        flat_votes = list(filter(prod_elected.__ne__, flat_votes))

    return total_elected, total_cost_elected


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--p', type=int, help='number of product', default=3)
    parser.add_argument('--C', type=list, help='cost per product', default=[1,1,1])
    parser.add_argument('--b', type=int, help='budget', default=2)
    parser.add_argument('--n', type=int, help='number of voters', default=3)
    parser.add_argument('--k', type=int, help='max number of products on a ballot', default=2)

    args = parser.parse_args()

    elected, cost_elected = greedy_approval(args.p, args.n, args.b, args.C, args.k)
    print(f'Elected {elected} with a total cost of {cost_elected}')
