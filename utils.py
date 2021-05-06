from itertools import chain, combinations


def k_set(prods, k):
    """Returns the set of possible combinations of products on a ballot with the maximum amount of
    products being k."""
    return chain.from_iterable(combinations(prods, n) for n in range(1, k + 1))
