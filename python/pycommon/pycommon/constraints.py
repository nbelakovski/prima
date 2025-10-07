# Project imports
from .linear_constraints import combine_multiple_linear_constraints
from .nonlinear_constraints import process_nl_constraints
# 3rd party imports
from scipy.optimize import NonlinearConstraint, LinearConstraint
# Standard library imports
from enum import Enum

class ConstraintType(Enum):
    LINEAR_OBJECT = 5
    NONLINEAR_OBJECT = 10
    LINEAR_DICT = 15
    NONLINEAR_DICT = 20


def get_constraint_type(constraint):
    if isinstance(constraint, dict) and ("A" in constraint) and ("lb" in constraint) and ("ub" in constraint):
        return ConstraintType.LINEAR_DICT
    elif isinstance(constraint, dict) and ("fun" in constraint) and ("lb" in constraint) and ("ub" in constraint):
        return ConstraintType.NONLINEAR_DICT
    elif hasattr(constraint, "A") and hasattr(constraint, "lb") and hasattr(constraint, "ub"):
        return ConstraintType.LINEAR_OBJECT
    elif hasattr(constraint, "fun") and hasattr(constraint, "lb") and hasattr(constraint, "ub"):
        return ConstraintType.NONLINEAR_OBJECT
    else:
        raise ValueError(f"Constraint type {type(constraint)} not recognized")


def process_constraints(constraints):
    # First throw it back if it's an empty tuple
    if not constraints:
        return None, None
    # Next figure out if it's a list of constraints or a single constraint
    # If it's a single constraint, make it a list, and then the remaining logic
    # doesn't have to change
    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]

    # Separate out the linear and nonlinear constraints
    linear_constraints = []
    nonlinear_constraints = []
    for constraint in constraints:
        constraint_type = get_constraint_type(constraint)
        if constraint_type is ConstraintType.LINEAR_OBJECT:
            linear_constraints.append(constraint)
        elif constraint_type is ConstraintType.NONLINEAR_OBJECT:
            nonlinear_constraints.append(constraint)
        elif constraint_type == ConstraintType.LINEAR_DICT:
            linear_constraints.append(LinearConstraint(constraint["A"], constraint["lb"], constraint["ub"]))
        elif constraint_type == ConstraintType.NONLINEAR_DICT:
            nonlinear_constraints.append(NonlinearConstraint(constraint["fun"], constraint["lb"], constraint["ub"]))
        else:
            raise ValueError("Constraint type not recognized")

    if len(nonlinear_constraints) > 0:
        nonlinear_constraint_function = process_nl_constraints(nonlinear_constraints)
    else:
        nonlinear_constraint_function = None

    # Determine if we have multiple linear constraints, just 1, or none, and process accordingly
    if len(linear_constraints) > 1:
        linear_constraint = combine_multiple_linear_constraints(linear_constraints)
    elif len(linear_constraints) == 1:
        linear_constraint = linear_constraints[0]
    else:
        linear_constraint = None

    return linear_constraint, nonlinear_constraint_function
