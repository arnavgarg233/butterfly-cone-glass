from __future__ import annotations

import math

import pytest
import torch

from butterfly_cone.engine.system import ParticleSystem
from butterfly_cone.rcce.slab import (
    SlabSelection,
    SlabSpec,
    resolve_divergence_components,
    select_slab,
    signed_offset_from_midplane,
)


def _system() -> ParticleSystem:
    # The midplane sits at z = 0 so that the particle at z = 5.90 straddles the
    # periodic seam: its minimum-image offset is -0.10, not +5.90.  A midplane
    # in the box interior would never exercise the fold.
    positions = torch.tensor(
        [
            [1.00, 1.00, 0.00],
            [1.00, 1.00, 0.40],
            [1.00, 1.00, 1.20],
            [1.00, 1.00, 5.90],
        ],
        dtype=torch.float64,
    )
    return ParticleSystem(
        positions=positions,
        velocities=torch.zeros_like(positions),
        diameters=torch.tensor([0.8, 0.9, 1.0, 1.1], dtype=torch.float64),
        box=torch.full((3,), 6.0, dtype=torch.float64),
        active_mask=torch.ones(4, dtype=torch.bool),
        unwrapped_positions=positions.clone(),
    )


def test_slab_selection_uses_minimum_image_and_labels_sub_layers() -> None:
    system = _system()
    spec = SlabSpec(axis=2, center=0.00, thickness=2.00, interface=0.50)

    offsets = signed_offset_from_midplane(system.positions, spec.axis, spec.center, system.box)
    selection = select_slab(system, spec)

    # the seam particle folds to -0.10, not +2.90
    torch.testing.assert_close(
        offsets, torch.tensor([0.0, 0.40, 1.20, -0.10], dtype=torch.float64)
    )
    # |offset| < 1.0 is mobile: particles 0, 1, 3.  particle 2 at 1.20 is wall.
    assert selection.mobile_indices.tolist() == [0, 1, 3]
    assert selection.wall_indices.tolist() == [2]
    # interface is the outer 0.5 of the mobile film, i.e. |offset| >= 0.5
    assert selection.interface_indices.tolist() == []
    assert selection.midfilm_indices.tolist() == [0, 1, 3]
    assert selection.n_mobile == 3
    assert selection.n_wall == 1
    assert selection.mobile_fraction == pytest.approx(0.75)


def test_interface_layer_separates_from_midfilm() -> None:
    system = _system()
    # widen the film so particle 2 (offset 1.20) becomes an interfacial mobile
    spec = SlabSpec(axis=2, center=0.00, thickness=3.00, interface=1.00)
    selection = select_slab(system, spec)

    assert selection.mobile_indices.tolist() == [0, 1, 2, 3]
    assert selection.wall_indices.tolist() == []
    # |offset| >= 0.5 is interfacial: 1.20 only (0.40 and 0.10 are mid-film)
    assert selection.interface_indices.tolist() == [2]
    assert selection.midfilm_indices.tolist() == [0, 1, 3]


def test_masks_are_a_partition_and_are_detached_copies() -> None:
    system = _system()
    spec = SlabSpec(axis=2, center=0.00, thickness=2.00, interface=0.50)
    selection = select_slab(system, spec)

    assert bool((selection.mobile_mask ^ selection.wall_mask).all())
    assert bool((selection.interface_mask | selection.midfilm_mask == selection.mobile_mask).all())
    assert not bool((selection.interface_mask & selection.midfilm_mask).any())
    # membership is frozen at parent time: moving the system must not change it
    system.positions[2, 2] = 0.00
    again = select_slab(system, spec)
    assert selection.mobile_indices.tolist() == [0, 1, 3]
    assert again.mobile_indices.tolist() == [0, 1, 2, 3]


def test_thickness_must_fit_inside_the_box() -> None:
    system = _system()
    with pytest.raises(ValueError, match="smaller than the box length"):
        select_slab(system, SlabSpec(axis=2, center=0.0, thickness=6.0, interface=0.0))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(axis=3, center=0.0, thickness=1.0, interface=0.0), "axis must be"),
        (dict(axis=2, center=0.0, thickness=0.0, interface=0.0), "thickness must be positive"),
        (dict(axis=2, center=0.0, thickness=1.0, interface=0.6), "interface must satisfy"),
        (dict(axis=2, center=0.0, thickness=1.0, interface=-0.1), "interface must satisfy"),
        (dict(axis=2, center=math.nan, thickness=1.0, interface=0.0), "must be finite"),
    ],
)
def test_spec_rejects_invalid_geometry(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SlabSpec(**kwargs)


def test_spec_round_trips_to_dict() -> None:
    spec = SlabSpec(axis=1, center=2.5, thickness=4.0, interface=1.0)
    assert spec.to_dict() == {
        "axis": 1,
        "center": 2.5,
        "thickness": 4.0,
        "interface": 1.0,
    }


def test_divergence_components_split_normal_from_in_plane() -> None:
    field = torch.tensor(
        [
            [3.0, 4.0, -5.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    normal, in_plane = resolve_divergence_components(field, axis=2)

    torch.testing.assert_close(normal, torch.tensor([5.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(in_plane, torch.tensor([5.0, 0.0], dtype=torch.float64))


def test_divergence_components_reject_bad_shape_and_axis() -> None:
    with pytest.raises(ValueError, match="trailing axis of size 3"):
        resolve_divergence_components(torch.zeros((2, 2), dtype=torch.float64), axis=2)
    with pytest.raises(ValueError, match="axis must be"):
        resolve_divergence_components(torch.zeros((2, 3), dtype=torch.float64), axis=7)


def test_selection_is_the_documented_type() -> None:
    system = _system()
    selection = select_slab(system, SlabSpec(axis=2, center=0.0, thickness=2.0, interface=0.5))
    assert isinstance(selection, SlabSelection)


def test_pinned_wall_freezes_the_wall_and_still_forces_the_film() -> None:
    """The engine already implements pinning: ``active_mask`` gates motion.

    Confinement therefore needs no integrator change.  This asserts the three
    properties the protocol depends on: wall particles do not move, film
    particles do, and the wall exerts force on the film (so the confined film
    is not merely an isolated sub-box).
    """

    from butterfly_cone.engine.integrate import MDIntegrator
    from butterfly_cone.engine.system import make_generator, make_system

    def run(freeze_wall: bool, steps: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
        system = make_system(
            512, generator=make_generator(11), density=1.0, dtype=torch.float64
        )
        box_z = float(system.box[2])
        spec = SlabSpec(axis=2, center=0.5 * box_z, thickness=0.5 * box_z, interface=1.0)
        selection = select_slab(system, spec)
        if freeze_wall:
            system.active_mask = selection.mobile_mask.clone()
        start = system.positions.clone()
        MDIntegrator(system, dt=0.002).step(steps)
        return system.positions - start, selection.mobile_mask

    confined_shift, mobile = run(freeze_wall=True)
    free_shift, _ = run(freeze_wall=False)

    wall = ~mobile
    assert int(mobile.sum()) > 0 and int(wall.sum()) > 0

    # 1. the pinned wall does not move, exactly
    torch.testing.assert_close(
        confined_shift[wall], torch.zeros_like(confined_shift[wall])
    )
    # 2. the mobile film does move
    assert float(confined_shift[mobile].abs().max()) > 0.0
    # 3. pinning changes the film's trajectory, so the wall is exerting force
    #    on it rather than the film evolving as an independent sub-system
    assert not torch.allclose(confined_shift[mobile], free_shift[mobile])


def test_anisotropy_ratio_is_calibrated_to_one_on_an_isotropic_field() -> None:
    """The calibration the first estimator lacked.

    An isotropic Gaussian displacement field must return exactly 1.  A
    mean-of-norms estimator normalised by sqrt(2) instead returns
    (pi/2)/sqrt(2) = 1.1107, which is the offset that showed up as a bulk
    control reading 1.099 rather than 1.000.
    """

    from butterfly_cone.rcce.slab import anisotropy_ratio

    generator = torch.Generator().manual_seed(7)
    field = torch.randn((200_000, 3), generator=generator, dtype=torch.float64)

    assert anisotropy_ratio(field, axis=2) == pytest.approx(1.0, abs=5e-3)

    # the discredited mean-of-norms route, shown to carry the pi/2 offset
    normal, in_plane = resolve_divergence_components(field, axis=2)
    naive = float(in_plane.mean() / normal.mean()) / math.sqrt(2.0)
    assert naive == pytest.approx(math.pi / 2.0 / math.sqrt(2.0), abs=5e-3)


def test_anisotropy_ratio_detects_a_squashed_normal_component() -> None:
    from butterfly_cone.rcce.slab import anisotropy_ratio

    generator = torch.Generator().manual_seed(11)
    field = torch.randn((100_000, 3), generator=generator, dtype=torch.float64)
    field[:, 2] *= 0.5  # suppress motion normal to the film

    # in-plane now carries 2 units of variance against 0.25 normal
    assert anisotropy_ratio(field, axis=2) == pytest.approx(2.0, abs=2e-2)
