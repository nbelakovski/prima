import numpy as np
try:
    from prima import LinearConstraint, combine_multiple_linear_constraints, separate_LC_into_eq_and_ineq, minimize
except ModuleNotFoundError:
    try:
        from pyprima import LinearConstraint, combine_multiple_linear_constraints, separate_LC_into_eq_and_ineq, minimize
    except ModuleNotFoundError:
        pytest.fail(reason="Could not find either prima or pyprima libraries")
from objective import fun


def test_multiple_linear_constraints_implementation():
    constraints = [LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=[5, 6], ub=[7, 8]),
                   LinearConstraint(A=np.array([[9, 10], [11, 12]]), lb=[13, 14], ub=[15, 16])]
    processed_constraint = combine_multiple_linear_constraints(constraints)
    assert (processed_constraint.A == np.array([[1, 2], [3, 4], [9, 10], [11, 12]])).all()
    assert all(processed_constraint.lb == [5, 6, 13, 14])
    assert all(processed_constraint.ub == [7, 8, 15, 16])
    

def test_multiple_linear_constraints_high_level():
    constraints = [LinearConstraint(A=np.array([1, 0]), lb=5, ub=7),
                     LinearConstraint(A=np.array([0, 1]), lb=6, ub=8)]
    x0 = [0.0, 0.0]
    res = minimize(fun, x0, constraints=constraints)
    assert np.isclose(res.x[0], 5.0, rtol=1e-6)
    assert np.isclose(res.x[1], 6.0, rtol=1e-6)
    if hasattr(res, 'fun'):
        assert np.isclose(res.fun, 4.0, rtol=1e-6)
    elif hasattr(res, 'f'):
        assert np.isclose(res.f, 4.0, rtol=1e-6)
    else:
        raise Exception("Result has no attribute for function value")


def test_separate_LC_into_eq_and_ineq():
    linear_constraint = LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=[5, 6], ub=[5, 8])
    A_eq, b_eq, A_ineq, b_ineq = separate_LC_into_eq_and_ineq(linear_constraint)
    assert (A_eq == np.array([[1, 2]])).all()
    assert all(b_eq == [5])
    assert (A_ineq == np.array([[-3, -4], [3, 4]])).all()
    assert all(b_ineq == [-6, 8])
