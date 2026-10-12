import jax
import jax.numpy as jnp

from applications.glassy_polymer.constitutive import (
    ConstitutiveRegularization,
    initial_state,
    pc_parameters,
    pmma_parameters,
    running_rate_average,
    spd_log,
    spd_power,
    update_material,
)
from applications.glassy_polymer.protocols import isochoric_uniaxial_F, run_strain_history
from applications.glassy_polymer.protocols import run_stress_history


def test_initial_state_is_stress_free():
    params = pc_parameters()
    state = initial_state(params)
    stress, updated, diag = update_material(jnp.eye(3), state, 0.01, 0.01, params)
    assert jnp.max(jnp.abs(stress)) < 1.0e-10
    assert jnp.max(jnp.abs(updated.Fp - jnp.eye(3))) < 1.0e-10
    assert jnp.isfinite(diag.residual)


def test_short_history_is_finite_and_differentiable():
    params = pc_parameters()
    state = initial_state(params)
    history = jnp.linspace(0.0, 0.03, 8)
    result = run_strain_history(history, 0.1, params, state)
    assert jnp.all(jnp.isfinite(result.stress))

    def loss(E):
        varied = params._replace(E=E)
        out = run_strain_history(history, 0.1, varied, initial_state(varied))
        return out.stress[-1, 1, 1]

    assert jnp.isfinite(jax.grad(loss)(params.E))


def test_compression_deformation_gradient_has_unit_jacobian():
    F = isochoric_uniaxial_F(jnp.asarray(0.8))
    assert jnp.isclose(jnp.linalg.det(F), 1.0, rtol=1.0e-12, atol=1.0e-12)


def test_traction_free_constraint_and_gradient():
    params = pc_parameters()
    history = jnp.linspace(0.0, 0.02, 4)

    def terminal_stress(E):
        varied = params._replace(E=E)
        result = run_strain_history(
            history, 0.1, varied, initial_state(varied), traction_free=True
        )
        assert jnp.max(jnp.abs(result.stress[:, 0, 0])) < 1.0e-8
        return -result.stress[-1, 1, 1]

    assert jnp.isfinite(jax.grad(terminal_stress)(params.E))


def test_vumat_path_is_finite():
    params = pc_parameters()
    result = run_strain_history(
        jnp.linspace(0.0, 0.05, 8), 0.2, params, initial_state(params),
        traction_free=True, substeps=5, scheme="vumat",
    )
    assert jnp.all(jnp.isfinite(result.stress))


def test_coupled_stress_control_enforces_targets():
    params = pc_parameters()
    target = jnp.full(4, 30.0)
    result = run_stress_history(
        target, jnp.full(4, 0.5), params, initial_state(params), substeps=5
    )
    assert jnp.max(jnp.abs(-result.stress[:, 1, 1] - target)) < 1.0e-8
    assert jnp.max(jnp.abs(result.stress[:, 0, 0])) < 1.0e-8


def test_nested_implicit_stress_control_parameter_gradient_is_finite():
    params = pmma_parameters()
    regularization = ConstitutiveRegularization(0.02, 4.0e-7)

    def terminal_creep(log_gamma0):
        varied = params._replace(gamma0=params.gamma0 * jnp.exp(log_gamma0))
        result = run_stress_history(
            jnp.full(5, 42.0), jnp.full(5, 0.5), varied,
            initial_state(varied), scheme="umat", substeps=4,
            regularization=regularization, checkpoint=True,
        )
        return result.strain[-1]

    derivative = jax.grad(terminal_creep)(jnp.asarray(0.0))
    assert jnp.isfinite(derivative)


def test_regularized_switch_converges_to_exact_response():
    params = pmma_parameters()
    strain = jnp.linspace(0.0, 0.12, 13)
    dt = jnp.full(strain.shape, 10.0)

    def prediction(width):
        result = run_strain_history(
            strain, dt, params, initial_state(params), traction_free=True,
            scheme="vumat", substeps=4,
            regularization=ConstitutiveRegularization(width, 0.0),
        )
        return result.stress[:, 1, 1]

    exact = prediction(0.0)
    coarse = jnp.linalg.norm(prediction(0.2) - exact)
    fine = jnp.linalg.norm(prediction(0.02) - exact)
    assert float(fine) < float(coarse)


def test_checkpointed_and_direct_history_gradients_match():
    params = pmma_parameters()
    strain = jnp.linspace(0.0, 0.12, 13)
    dt = jnp.full(strain.shape, 10.0)
    regularization = ConstitutiveRegularization(0.02, 4.0e-7)

    def objective(log_s0, checkpoint):
        local = params._replace(s0=jnp.exp(log_s0))
        result = run_strain_history(
            strain, dt, local, initial_state(local), traction_free=True,
            scheme="vumat", substeps=4, regularization=regularization,
            checkpoint=checkpoint,
        )
        return jnp.mean(result.stress[:, 1, 1] ** 2)

    direct = jax.grad(lambda value: objective(value, False))(jnp.log(params.s0))
    checkpointed = jax.grad(lambda value: objective(value, True))(jnp.log(params.s0))
    assert jnp.isfinite(direct)
    assert jnp.allclose(direct, checkpointed, rtol=1.0e-9, atol=1.0e-9)


def test_spd_log_jvp_is_finite_at_repeated_eigenvalues():
    tensor = jnp.eye(3)
    direction = jnp.asarray([[0.2, 0.1, 0.0], [0.1, -0.1, 0.04], [0.0, 0.04, 0.3]])
    _, tangent = jax.jvp(spd_log, (tensor,), (direction,))
    step = 1.0e-5
    finite_difference = (spd_log(tensor + step * direction)
                         - spd_log(tensor - step * direction)) / (2.0 * step)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.allclose(tangent, finite_difference, rtol=2.0e-8, atol=2.0e-8)


def test_spd_log_jvp_remains_accurate_for_nearly_repeated_roots():
    direction = jnp.asarray([[0.2, 0.07, -0.03], [0.07, -0.1, 0.05],
                             [-0.03, 0.05, 0.3]])
    for gap in (1.0e-2, 1.0e-5, 1.0e-8, 1.0e-12):
        tensor = jnp.diag(jnp.asarray([1.0, 1.0 + gap, 2.0]))
        _, tangent = jax.jvp(spd_log, (tensor,), (direction,))
        step = 2.0e-6
        finite_difference = (spd_log(tensor + step * direction)
                             - spd_log(tensor - step * direction)) / (2.0 * step)
        relative = jnp.linalg.norm(tangent - finite_difference) / jnp.linalg.norm(finite_difference)
        assert relative < 2.0e-8


def test_spd_log_jvp_is_rotation_covariant_at_repeated_roots():
    angle = 0.43
    rotation = jnp.asarray([
        [jnp.cos(angle), -jnp.sin(angle), 0.0],
        [jnp.sin(angle), jnp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    tensor = jnp.diag(jnp.asarray([1.7, 1.7, 3.1]))
    direction = jnp.asarray([[0.2, 0.08, 0.02], [0.08, -0.1, -0.04],
                             [0.02, -0.04, 0.3]])
    _, tangent = jax.jvp(spd_log, (tensor,), (direction,))
    rotated_tensor = rotation @ tensor @ rotation.T
    rotated_direction = rotation @ direction @ rotation.T
    _, rotated_tangent = jax.jvp(
        spd_log, (rotated_tensor,), (rotated_direction,)
    )
    assert jnp.allclose(rotated_tangent, rotation @ tangent @ rotation.T,
                        rtol=2.0e-10, atol=2.0e-10)


def _power_directional_error(tensor, direction, power, power_direction=0.0,
                             step=2.0e-6):
    _, tangent = jax.jvp(
        spd_power, (tensor, jnp.asarray(power)),
        (direction, jnp.asarray(power_direction)),
    )
    finite_difference = (
        spd_power(tensor + step * direction, power + step * power_direction)
        - spd_power(tensor - step * direction, power - step * power_direction)
    ) / (2.0 * step)
    denominator = jnp.maximum(jnp.linalg.norm(finite_difference), 1.0e-14)
    return jnp.linalg.norm(tangent - finite_difference) / denominator


def test_spd_power_jvp_covers_repeated_near_repeated_and_random_spectra():
    direction = jnp.asarray([
        [0.12, 0.03, -0.02], [0.03, -0.08, 0.04], [-0.02, 0.04, 0.16]
    ])
    angle = 0.41
    rotation = jnp.asarray([
        [jnp.cos(angle), -jnp.sin(angle), 0.0],
        [jnp.sin(angle), jnp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    spectra = (
        jnp.asarray([1.0, 1.0, 1.0]),
        jnp.asarray([0.8, 0.8 + 1.0e-10, 2.1]),
        jnp.asarray([0.63, 1.27, 2.42]),
    )
    tensors = tuple(rotation @ jnp.diag(values) @ rotation.T for values in spectra)
    for power in (0.2, 0.5, 0.9, 1.0):
        for tensor in tensors:
            error = _power_directional_error(tensor, direction, power)
            assert error < 2.0e-8


def test_spd_power_exponent_jvp_matches_centered_difference():
    tensor = jnp.asarray([
        [1.4, 0.13, -0.04], [0.13, 0.92, 0.07], [-0.04, 0.07, 1.8]
    ])
    zero = jnp.zeros((3, 3), dtype=jnp.float64)
    for power in (0.2, 0.5, 0.9, 1.0):
        error = _power_directional_error(
            tensor, zero, power, power_direction=0.37, step=1.0e-6
        )
        assert error < 2.0e-9


def test_spd_power_jvp_is_rotation_covariant():
    angle = 0.73
    rotation = jnp.asarray([
        [jnp.cos(angle), 0.0, jnp.sin(angle)],
        [0.0, 1.0, 0.0],
        [-jnp.sin(angle), 0.0, jnp.cos(angle)],
    ])
    tensor = jnp.diag(jnp.asarray([0.9, 0.9, 2.3]))
    direction = jnp.asarray([
        [0.1, 0.04, -0.02], [0.04, -0.08, 0.03], [-0.02, 0.03, 0.14]
    ])
    _, tangent = jax.jvp(
        spd_power, (tensor, jnp.asarray(0.5)),
        (direction, jnp.asarray(0.17)),
    )
    _, rotated_tangent = jax.jvp(
        spd_power,
        (rotation @ tensor @ rotation.T, jnp.asarray(0.5)),
        (rotation @ direction @ rotation.T, jnp.asarray(0.17)),
    )
    assert jnp.allclose(
        rotated_tangent, rotation @ tangent @ rotation.T,
        rtol=2.0e-10, atol=2.0e-10,
    )


def test_zero_driving_state_freezes_common_inelastic_map():
    params = pmma_parameters()
    state = initial_state(params)
    _, updated, diagnostics = update_material(
        jnp.eye(3), state, jnp.asarray(0.25), jnp.asarray(0.25), params
    )
    assert diagnostics.q0 == 0.0
    assert jnp.array_equal(updated.Fp, state.Fp)


def test_recoverable_branch_advances_common_map_when_primary_branch_is_idle():
    params = pmma_parameters()
    state = initial_state(params)._replace(
        A1=jnp.diag(jnp.exp(jnp.asarray([0.01, -0.02, 0.01])))
    )
    _, updated, diagnostics = update_material(
        jnp.eye(3), state, jnp.asarray(0.1), jnp.asarray(0.1), params
    )
    assert diagnostics.q0 == 0.0
    assert diagnostics.q1 > 0.0
    assert not jnp.array_equal(updated.Fp, state.Fp)


def test_hydrostatic_tension_uses_declared_zero_direction_rule():
    params = pmma_parameters()
    state = initial_state(params)
    deformation = 1.01 * jnp.eye(3)
    _, updated, diagnostics = update_material(
        deformation, state, jnp.asarray(0.25), jnp.asarray(0.25), params
    )
    assert diagnostics.tau0 == 0.0
    assert diagnostics.q0 == 0.0
    assert jnp.array_equal(updated.Fp, state.Fp)


def test_stabilized_structure_tensors_remain_spd_for_a_coarse_active_step():
    params = pmma_parameters()
    state = initial_state(params)
    deformation = isochoric_uniaxial_F(jnp.asarray(0.12))
    _, updated, _ = update_material(
        deformation, state, jnp.asarray(20.0), jnp.asarray(20.0), params
    )
    for tensor in (updated.A0, updated.A1, updated.A2):
        assert jnp.min(jnp.linalg.eigvalsh(tensor)) > 0.0


def test_zero_increment_does_not_advance_internal_state():
    params = pmma_parameters()
    state = initial_state(params)
    _, updated, _ = update_material(
        jnp.diag(jnp.asarray([1.01, 0.98, 1.01])), state,
        jnp.asarray(0.0), jnp.asarray(0.0), params,
    )
    assert jnp.array_equal(updated.Fp, state.Fp)
    assert jnp.array_equal(updated.A0, state.A0)
    assert jnp.array_equal(updated.A1, state.A1)
    assert jnp.array_equal(updated.A2, state.A2)
    assert updated.gamma_p == state.gamma_p
    assert updated.time == state.time


def test_running_rate_average_has_exact_first_step_and_semigroup_property():
    rate = jnp.asarray(0.037)
    epsilon = 1.0e-5
    whole = running_rate_average(0.0, rate, 1.0, 0.0, epsilon)
    first = running_rate_average(0.0, rate, 0.2, 0.0, epsilon)
    split = running_rate_average(first, rate, 0.8, 0.2, epsilon)
    assert jnp.isfinite(whole)
    assert jnp.allclose(whole, rate / (1.0 + epsilon), rtol=1.0e-14, atol=1.0e-14)
    assert jnp.allclose(split, whole, rtol=1.0e-14, atol=1.0e-14)


def test_restart_uses_clock_stored_in_material_state():
    params = pmma_parameters()
    full_strain = jnp.asarray([0.0, 0.01, 0.02, 0.03, 0.04])
    full = run_strain_history(
        full_strain, jnp.full(5, 0.1), params, initial_state(params), substeps=4
    )
    first = run_strain_history(
        full_strain[:3], jnp.full(3, 0.1), params, initial_state(params), substeps=4
    )
    state = jax.tree_util.tree_map(lambda value: value[-1], first.state)
    second = run_strain_history(
        full_strain[3:], jnp.full(2, 0.1), params, state, substeps=4,
        previous_strain0=float(full_strain[2]),
    )
    assert jnp.allclose(second.stress, full.stress[3:], rtol=1.0e-11, atol=1.0e-11)
    for restarted, continuous in zip(second.state, full.state):
        assert jnp.allclose(restarted, continuous[3:], rtol=1.0e-11, atol=1.0e-11)


def test_one_step_cauchy_stress_is_objective_from_virgin_state():
    params = pmma_parameters()
    state = initial_state(params)
    angle = 0.37
    rotation = jnp.asarray([
        [jnp.cos(angle), -jnp.sin(angle), 0.0],
        [jnp.sin(angle), jnp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    deformation = jnp.diag(jnp.asarray([1.03, 0.94, 1.02]))
    stress, _, _ = update_material(deformation, state, 0.2, 0.2, params)
    rotated_stress, _, _ = update_material(rotation @ deformation, state, 0.2, 0.2, params)
    assert jnp.allclose(rotated_stress, rotation @ stress @ rotation.T,
                        rtol=2.0e-9, atol=2.0e-9)


def test_objectivity_from_anisotropic_preconditioned_state():
    params = pmma_parameters()
    strain = jnp.asarray([0.0, 0.04, 0.08, 0.04])
    prepared = run_strain_history(
        strain, jnp.full(strain.shape, 0.02), params, initial_state(params),
        traction_free=False, substeps=8,
    )
    state = jax.tree_util.tree_map(lambda value: value[-1], prepared.state)
    assert jnp.linalg.norm(state.A0 - jnp.eye(3)) > 1.0e-6

    angle = 0.51
    rotation = jnp.asarray([
        [jnp.cos(angle), -jnp.sin(angle), 0.0],
        [jnp.sin(angle), jnp.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    deformation = jnp.asarray([
        [1.02, 0.015, 0.0], [0.0, 0.965, 0.012], [0.0, 0.0, 1.016]
    ])
    stress, updated, _ = update_material(deformation, state, 0.17, 0.17, params)
    rotated_state = state._replace(F=rotation @ state.F)
    rotated_stress, rotated, _ = update_material(
        rotation @ deformation, rotated_state, 0.17, 0.17, params
    )
    assert jnp.allclose(
        rotated_stress, rotation @ stress @ rotation.T,
        rtol=3.0e-9, atol=3.0e-9,
    )
    # Fp and A_i are stored in the material/intermediate frame and therefore
    # remain invariant under a superposed spatial rotation; only F covaries.
    assert jnp.allclose(rotated.F, rotation @ updated.F, rtol=1.0e-12, atol=1.0e-12)
    for original, transformed in zip(
        (updated.Fp, updated.A0, updated.A1, updated.A2),
        (rotated.Fp, rotated.A0, rotated.A1, rotated.A2),
    ):
        assert jnp.allclose(transformed, original, rtol=3.0e-9, atol=3.0e-9)


def test_isochoric_tensor_updates_preserve_unit_determinants():
    params = pmma_parameters()
    history = jnp.asarray([0.0, 0.04, 0.08, 0.04, 0.0])
    result = run_strain_history(
        history, jnp.full(history.shape, 0.02), params, initial_state(params),
        traction_free=False, substeps=8,
    )
    for tensor_history in (result.state.Fp, result.state.A0,
                           result.state.A1, result.state.A2):
        determinants = jax.vmap(jnp.linalg.det)(tensor_history)
        assert jnp.allclose(determinants, 1.0, rtol=2.0e-10, atol=2.0e-10)
