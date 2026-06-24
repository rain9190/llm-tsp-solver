import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def generate_instance(n_cities, seed=None):
    """Return n random city coordinates in the unit square as an (n, 2) array."""
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2))

def tour_length(coords, tour):
    """Total Euclidean length of the closed tour given by index list `tour`."""
    pts = coords[tour]
    nxt = np.roll(pts, -1, axis=0) # each city's successor in the tour
    return float(np.sqrt(((pts - nxt) ** 2).sum(axis=1)).sum())

def solve_with_ortools(coords, time_limit_ms=100):
    """Return a near-optimal tour (list of city indices) using OR-Tools routing."""
    n = len(coords)
    scale = 100000 # OR-Tools needs integer distances
    dist = (np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)) * scale).astype(int)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0) # n nodes, 1 vehicle, start=0
    routing = pywrapcp.RoutingModel(manager)

    def cb(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromMilliseconds(time_limit_ms)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None
    tour, node = [], routing.Start(0)
    while not routing.IsEnd(node):
        tour.append(manager.IndexToNode(node))
        node = sol.Value(routing.NextVar(node))
        return tour