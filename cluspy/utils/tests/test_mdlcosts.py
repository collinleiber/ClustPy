from cluspy.utils._mdlcosts import mdl_costs_bic, mdl_costs_discrete_probability, mdl_costs_orthogonal_matrix, \
    mdl_costs_integer_value


def test_mdl_costs_bic():
    costs = mdl_costs_bic(2.5)
    assert costs == 0.6609640474436812


def test_mdl_costs_discrete_probability():
    costs = mdl_costs_discrete_probability(1 / 33)
    assert abs(costs - 5.044394119) < 1e-9


def test_mdl_costs_orthogonal_matrix():
    n_points = 100
    dimenionality = 8
    costs = mdl_costs_orthogonal_matrix(dimenionality, mdl_costs_bic(n_points))
    assert costs == 28 * mdl_costs_bic(n_points)


def test_mdl_costs_integer_value():
    costs = mdl_costs_integer_value(77)
    assert abs(costs - 12.328150766) < 1e-9
