"""Parameter configuration for the ButterflyCone event-detection machinery.

IMPORTANT SCIENTIFIC CONSTRAINT (PLAN_v2.1 sec 8): every threshold and definition
in this package is provisional and gets FROZEN later in advance-declaration after the
pilot data are seen.  Nothing is hardcoded; every knob lives here as a dataclass
field with an explicit default.  These are instruments, not scientific choices.

All values are PROVISIONAL until the advance-declaration freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DisplacementConfig:
    """Cage-relative displacement and overlap parameters."""

    # First-coordination-shell neighbour cutoff, in units of the pair diameter
    # sigma_ij.  Neighbours of i are j with |r_ij| < first_shell_factor * sigma_ij.
    first_shell_factor: float = 1.4
    # Overlap indicator length scale a: w_i = 1 iff |plain displacement| < a.
    overlap_a: float = 0.3
    # Behaviour for particles that have no first-shell neighbours at the
    # reference frame: "plain" leaves the cage-relative displacement equal to the
    # plain displacement (empty-neighbour-mean limit); "zero" sets it to zero.
    isolated_fallback: str = "plain"


@dataclass(frozen=True)
class HopConfig:
    """Rearrangement-detector parameters for both interchangeable detectors."""

    # --- shared ---
    first_shell_factor: float = 1.4
    reference_frame: int = 0

    # --- persistent-displacement detector ---
    # Threshold a on the (cage-relative by default) displacement magnitude.
    persistent_threshold: float = 0.3
    # Sustained-motion window, in trajectory time units.
    persist_time: float = 5.0
    # Operate on cage-relative displacement (True, the spec default) or the plain
    # displacement (False).
    persistent_cage_relative: bool = True

    # --- p_hop-style (Candelier) detector ---
    # Total two-window width in trajectory time units; each half-window is
    # phop_window_time / 2 wide.
    phop_window_time: float = 10.0
    # Threshold on the p_hop statistic (units of length^2).
    phop_threshold: float = 0.1
    # p_hop on cage-relative (True) or plain/absolute (False, Candelier default).
    phop_cage_relative: bool = False


@dataclass(frozen=True)
class ClusterConfig:
    """Spatiotemporal clustering and event-persistence parameters."""

    # Two rearranging particles are linked when their onset times differ by less
    # than dt_link (time units) AND their onset positions are within r_link.
    dt_link: float = 5.0
    # Link distance; default is the nominal first-shell distance.
    r_link: float = 1.4
    # Persistence check: an event is "persistent" if, at reversal_t_check after
    # its onset, the members' cage-relative displacement has not reversed below
    # reversal_fraction of its onset magnitude.
    reversal_t_check: float = 10.0
    reversal_fraction: float = 0.5
    first_shell_factor: float = 1.4
    reference_frame: int = 0


@dataclass(frozen=True)
class StringConfig:
    """Microstring follow-the-leader tracing parameters."""

    # A follower's displacement endpoint must land within r_string of a leader's
    # original position.
    r_string: float = 0.5
    # Lag window (time units) over which the follower displacement is measured.
    lag_time: float = 10.0


@dataclass(frozen=True)
class AttributionConfig:
    """Seed-versus-incoming convention parameters (PLAN_v2.1 sec 8)."""

    # --- A: first-nucleus / bond-breaking ---
    # A particle is "bond-breaking" when the fraction of its reference-frame
    # first-shell neighbours it has lost exceeds bond_break_loss_fraction,
    # sustained for bond_break_persist_time.
    bond_break_loss_fraction: float = 0.3
    bond_break_persist_time: float = 5.0

    # --- B: core-majority ---
    majority_theta: float = 0.5  # also exposed at 0.6 and 0.7 via the API

    # --- D: material-core dominance ---
    material_dominance: float = 0.6

    # Linking of bond-breaking particles into nuclei (convention A).
    nucleus_dt_link: float = 5.0
    nucleus_r_link: float = 1.4

    # Shared building blocks.
    first_shell_factor: float = 1.4
    reference_frame: int = 0


# Convenience thresholds explicitly exposed by the spec for convention B.
MAJORITY_THETAS: tuple[float, ...] = (0.5, 0.6, 0.7)


@dataclass(frozen=True)
class CavitySpec:
    """A monitored cavity: core sphere of radius R_core, monitoring annulus
    R_core..R_annulus, centred at ``center`` (a length-3 coordinate)."""

    center: tuple[float, float, float]
    r_core: float
    r_annulus: float

    def __post_init__(self) -> None:
        if self.r_core <= 0.0 or self.r_annulus < self.r_core:
            raise ValueError("require 0 < r_core <= r_annulus")
        if len(self.center) != 3:
            raise ValueError("center must be a length-3 coordinate")


@dataclass(frozen=True)
class EventConfig:
    """Bundle of every sub-config for end-to-end pipelines."""

    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    hop: HopConfig = field(default_factory=HopConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    string: StringConfig = field(default_factory=StringConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
