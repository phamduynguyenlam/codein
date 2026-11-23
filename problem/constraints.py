import numpy as np


def evaluate_objectives_and_constraints(problem, x):
    """Return aligned objective and constraint matrices, including [N, 0] for G."""
    x = np.atleast_2d(x)
    evaluate_joint = getattr(problem, "evaluate_with_constraints", None)
    if evaluate_joint is None:
        y = problem.evaluate(x)
        g = np.empty((len(x), 0), dtype=np.float32)
    else:
        y, g = evaluate_joint(x)
        g = np.asarray(g, dtype=np.float32)
        if g.ndim == 1:
            g = g.reshape(len(x), -1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or g.ndim != 2 or len(y) != len(x) or len(g) != len(x):
        raise ValueError("The problem must return objective and constraint matrices aligned with x.")
    return y, g


def initial_constraint_scale(g_initial, epsilon=1e-8):
    if g_initial.shape[1] == 0:
        return np.empty(0, dtype=np.float32)
    return np.maximum(np.max(np.abs(g_initial), axis=0), epsilon).astype(np.float32)
