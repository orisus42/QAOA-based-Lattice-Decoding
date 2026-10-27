"""Public-data-only classical controls for the Module-LWE experiment."""
from itertools import combinations, product
import numpy as np


def local_refine(B, target, start, max_changed=1, budget=1024):
    """Greedy integer coordinate moves; stop at a local minimum or evaluation cap.

Consider all +/-1 moves affecting at most max_changed coordinates each round.
Keep the incumbent, accept strict cost reductions, and break ties by move order.
The domain is the full lattice, not restricted to the binary box.
"""
    if max_changed not in [1, 2] or budget < 1:
        raise ValueError("use one/two-coordinate moves and a positive budget")
    B, target, best = np.asarray(B), np.asarray(target), np.array(start, dtype=np.int64)
    moves = []
    for changed in range(1, max_changed+1):
        for indices in combinations(range(len(best)), changed):
            for signs in product([-1, 1], repeat=changed):
                move = np.zeros(len(best), dtype=np.int64)
                move[list(indices)] = signs
                moves.append(move)
    moves = np.array(moves)
    best_cost = int(np.sum((B@best-target)**2))
    evaluations, steps = 1, 0
    while evaluations+len(moves) <= budget:
        trials = best+moves
        costs = np.sum((trials@B.T-target)**2, axis=1)
        evaluations += len(moves)
        winner = int(costs.argmin())
        if costs[winner] >= best_cost:
            return {"coefficients": best.tolist(), "cost": best_cost, "evaluations": evaluations,
                    "steps": steps, "stopping_reason": "local_minimum"}
        best, best_cost = trials[winner], int(costs[winner])
        steps += 1
    return {"coefficients": best.tolist(), "cost": best_cost, "evaluations": evaluations,
            "steps": steps, "stopping_reason": "evaluation_budget"}


def lift_secrets(A, target, q, secrets):
    """Best modular lifts for candidate secrets, scored by the same CVP cost."""
    A, target, secrets = np.asarray(A), np.asarray(target), np.asarray(secrets, dtype=np.int64)
    wraps = (target-secrets@A.T+q//2)//q
    residual = q*wraps+secrets@A.T-target
    costs = np.sum(residual**2, axis=1)+np.sum(secrets**2, axis=1)
    return np.concatenate([wraps, secrets], axis=1), costs
