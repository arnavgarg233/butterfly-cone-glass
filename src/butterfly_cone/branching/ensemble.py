"""Append-only execution and publication of momentum-branch ensembles.

Two frame-capture modes are supported:

* **full capture** (default) -- every captured frame is retained and the whole
  ``(T, B, N, 3)`` trajectory is stacked in memory and published per branch.
  Correct and convenient for short horizons.
* **streaming capture** -- when a :class:`FrameReducer` is supplied, frames are
  fed to the reducer one at a time and never accumulated, so peak memory is
  ``O(B * N)`` instead of ``O(T * B * N)``.  The O1 observational pilot at
  horizon 1e4 / stride 1 is ~43 GB of frames and cannot be held on a 48 GB
  Apple-silicon host; the streaming path makes that horizon runnable by keeping only the
  reducer's online statistic (e.g. running peak per-particle displacement) plus
  the current frame.  ``max_frames`` additionally caps the number of *captured*
  frames (thinning the strided schedule) in either mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from typing import Any

import torch

from butterfly_cone.engine.system import ParticleSystem, make_generator
from butterfly_cone.harness.runs import RunManager

from .batched import (
    BatchedBussiThermostat,
    BatchedMDIntegrator,
    BatchedSystem,
    branch_maxwell_boltzmann_velocities,
)


TORCH_SEED_MODULUS = 2**63 - 1


def torch_seed(issued_seed: int) -> int:
    """Project a full harness SHA-256 integer into PyTorch's seed range."""

    if isinstance(issued_seed, bool) or not isinstance(issued_seed, int):
        raise TypeError("issued_seed must be an integer")
    return int(issued_seed) % TORCH_SEED_MODULUS


@dataclass(frozen=True)
class BatchedTrajectory:
    """CPU copies of branch trajectory frames at documented integer steps."""

    steps: tuple[int, ...]
    positions: torch.Tensor
    unwrapped_positions: torch.Tensor
    velocities: torch.Tensor


@dataclass(frozen=True)
class StreamingFrame:
    """One captured branch frame handed to a :class:`FrameReducer`.

    All tensors are detached CPU copies with shape ``(B, N, 3)``.  The frame is
    transient: it is discarded as soon as ``FrameReducer.update`` returns, so a
    reducer that keeps a growing per-frame history defeats the streaming cap.
    """

    frame_index: int
    step: int
    positions: torch.Tensor
    unwrapped_positions: torch.Tensor
    velocities: torch.Tensor


class FrameReducer:
    """Online, memory-bounded reduction of streamed branch frames.

    ``begin`` runs once before the first frame with the ensemble geometry;
    ``update`` runs once per captured frame in increasing-step order and MUST
    NOT retain a growing per-frame history; ``result`` returns the finalized
    reduction after the last frame.  Subclass and override ``update``/``result``.
    """

    def begin(
        self,
        *,
        count: int,
        n_particles: int,
        capture_steps: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:  # noqa: D401 - hook, default no-op
        """Initialize accumulators for a ``count``-branch, ``n_particles`` run."""

    def update(self, frame: StreamingFrame) -> None:
        raise NotImplementedError("FrameReducer subclasses must implement update()")

    def result(self) -> Any:
        raise NotImplementedError("FrameReducer subclasses must implement result()")


@dataclass
class PeakDisplacementResult:
    """Online rearrangement-relevant summary of a streamed branch ensemble.

    ``peak_displacement`` is the running max over captured frames of the
    per-(branch, particle) unwrapped displacement magnitude from frame 0 -- the
    core quantity a rearrangement/fate detector needs -- and the ``final_*``
    tensors are the last captured frame.  Every field is ``O(B * N)``, never
    ``O(T * B * N)``.
    """

    reference_unwrapped: torch.Tensor  # (B, N, 3) frame-0 unwrapped positions
    final_positions: torch.Tensor  # (B, N, 3)
    final_unwrapped: torch.Tensor  # (B, N, 3)
    final_velocities: torch.Tensor  # (B, N, 3)
    peak_displacement: torch.Tensor  # (B, N) running max |u(t) - u(0)|
    n_frames: int
    steps: tuple[int, ...]


class PeakDisplacementReducer(FrameReducer):
    """Running peak per-particle unwrapped displacement + the final frame.

    Exact (running max is associative), so the streamed statistic is
    bitwise-identical to the same reduction over a retained full trajectory.
    Memory footprint is one reference frame, one ``(B, N)`` peak buffer, and the
    current frame -- independent of the horizon.
    """

    def __init__(self) -> None:
        self._steps: tuple[int, ...] = ()
        self._reference: torch.Tensor | None = None
        self._peak: torch.Tensor | None = None
        self._final: StreamingFrame | None = None
        self._n_frames = 0

    def begin(self, *, count, n_particles, capture_steps, device, dtype) -> None:
        self._steps = tuple(capture_steps)
        self._reference = None
        self._peak = None
        self._final = None
        self._n_frames = 0

    def update(self, frame: StreamingFrame) -> None:
        unwrapped = frame.unwrapped_positions
        if self._reference is None:
            self._reference = unwrapped.clone()
            self._peak = torch.zeros(
                unwrapped.shape[0], unwrapped.shape[1], dtype=unwrapped.dtype
            )
        magnitude = torch.linalg.vector_norm(unwrapped - self._reference, dim=2)
        assert self._peak is not None
        torch.maximum(self._peak, magnitude, out=self._peak)
        self._final = frame
        self._n_frames += 1

    def result(self) -> PeakDisplacementResult:
        if self._final is None or self._reference is None or self._peak is None:
            raise RuntimeError("no frames were streamed to the reducer")
        return PeakDisplacementResult(
            reference_unwrapped=self._reference,
            final_positions=self._final.positions,
            final_unwrapped=self._final.unwrapped_positions,
            final_velocities=self._final.velocities,
            peak_displacement=self._peak,
            n_frames=self._n_frames,
            steps=self._steps,
        )


def _capture_steps(horizon: int, stride: int, max_frames: int | None) -> tuple[int, ...]:
    """Integer steps at which to capture a frame, always including 0 and horizon.

    The base schedule is every ``stride`` steps (with ``horizon`` appended when
    it is not already a multiple), reproducing the historical strided capture.
    When ``max_frames`` is smaller than that schedule, the schedule is thinned
    to roughly-even indices -- keeping the first and last -- so at most
    ``max_frames`` frames are ever captured (and, in full mode, retained).
    """

    base = list(range(0, horizon + 1, stride))
    if not base or base[-1] != horizon:
        base.append(horizon)
    if max_frames is None or len(base) <= max_frames:
        return tuple(base)
    if max_frames <= 1:
        return (horizon,) if horizon > 0 else (0,)
    if max_frames == 2:
        return (base[0], base[-1])
    last = len(base) - 1
    keep = sorted({round(index * last / (max_frames - 1)) for index in range(max_frames)})
    return tuple(base[index] for index in keep)


@dataclass(frozen=True)
class BranchEnsembleResult:
    """In-memory result handles corresponding to published run artifacts.

    ``trajectory`` is the retained ``(T, B, N, 3)`` full-capture trajectory; it
    is ``None`` in streaming mode, where ``reduction`` carries the reducer's
    online result instead.  ``capture_meta`` records the capture schedule and
    the peak number of simultaneously retained frames (1 while streaming).
    """

    run: RunManager
    branch_seeds: tuple[int, ...]
    branch_torch_seeds: tuple[int, ...]
    thermostat_seed: int | None
    thermostat_torch_seed: int | None
    final_states: tuple[ParticleSystem, ...]
    trajectory: BatchedTrajectory | None
    reduction: Any = None
    capture_meta: dict[str, Any] = field(default_factory=dict)


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def _state_payload(system: ParticleSystem) -> dict[str, torch.Tensor]:
    return {name: _cpu_tensor(value) for name, value in system.state_dict().items()}


def _save_bytes(value: Any) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _parent_state_sha256(parent: ParticleSystem) -> str:
    """Hash the complete frozen parent state that defines branch identity."""

    return hashlib.sha256(_save_bytes(_state_payload(parent))).hexdigest()


def _validate_controls(
    *,
    count: int,
    temperature: float,
    horizon: int,
    dt: float,
    stride: int,
    skin: float,
    thermostat_tau: float | None,
    max_frames: int | None,
) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if temperature < 0.0:
        raise ValueError("temperature must be nonnegative")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    if dt <= 0.0 or stride <= 0 or skin <= 0.0:
        raise ValueError("dt, stride, and skin must be positive")
    if thermostat_tau is not None and (thermostat_tau <= 0.0 or temperature <= 0.0):
        raise ValueError("Bussi NVT requires positive temperature and thermostat_tau")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer when supplied")


def run_branch_ensemble(
    parent: ParticleSystem,
    *,
    count: int,
    temperature: float,
    horizon: int,
    run: RunManager,
    dt: float = 0.01,
    stride: int = 1,
    skin: float = 0.3,
    thermostat_tau: float | None = None,
    parent_id: str | None = None,
    momentum_seed_domain: str = "branching.momentum",
    thermostat_seed_domain: str = "branching.thermostat",
    max_frames: int | None = None,
    frame_reducer: FrameReducer | None = None,
) -> BranchEnsembleResult:
    """Run and publish independent momentum branches from one frozen parent.

    ``run`` must be a newly created :class:`~butterfly_cone.harness.runs.RunManager`.
    Every momentum branch gets one ledger allocation.  When ``thermostat_tau``
    is supplied, a separate recorded allocation initializes the single
    branch-major Bussi stream.

    Frame capture has two modes.  By default (``frame_reducer is None``) the run
    is **full capture**: every strided frame is retained, the ``(T, B, N, 3)``
    trajectory is published per branch as a ``.pt`` file, and the final states
    are indexed by ``branch_provenance.json``.  When a :class:`FrameReducer` is
    supplied the run is **streaming**: frames are reduced online and never
    accumulated (peak memory ``O(B * N)``), no per-branch trajectory file is
    written, ``result.trajectory`` is ``None``, and ``result.reduction`` holds
    the reducer's finalized statistic -- this is what makes the O1 horizon
    (~43 GB of frames) runnable on a 48 GB Apple-silicon host.  ``max_frames`` caps the number
    of captured frames (thinning the strided schedule) in either mode.
    """

    _validate_controls(
        count=count,
        temperature=temperature,
        horizon=horizon,
        dt=dt,
        stride=stride,
        skin=skin,
        thermostat_tau=thermostat_tau,
        max_frames=max_frames,
    )
    if not isinstance(run, RunManager):
        raise TypeError("run must be a RunManager")
    if not momentum_seed_domain or not thermostat_seed_domain:
        raise ValueError("seed domains must be non-empty")
    if parent_id is not None and not parent_id:
        raise ValueError("parent_id must be non-empty when supplied")

    try:
        issued_branch_seeds = tuple(run.seed_for(momentum_seed_domain, index) for index in range(count))
        branch_torch_seeds = tuple(torch_seed(seed) for seed in issued_branch_seeds)
        system = BatchedSystem.from_system(parent, count)
        system.velocities = branch_maxwell_boltzmann_velocities(
            parent.n_particles,
            temperature,
            [make_generator(seed) for seed in branch_torch_seeds],
            device=parent.device,
            dtype=parent.dtype,
            active_mask=parent.active_mask,
        )

        thermostat_seed: int | None = None
        thermostat_torch_seed: int | None = None
        thermostat: BatchedBussiThermostat | None = None
        if thermostat_tau is not None:
            thermostat_seed = run.seed_for(thermostat_seed_domain, 0)
            thermostat_torch_seed = torch_seed(thermostat_seed)
            thermostat = BatchedBussiThermostat(
                temperature=temperature,
                tau=thermostat_tau,
                generator=make_generator(thermostat_torch_seed),
            )

        integrator = BatchedMDIntegrator(
            system,
            dt=dt,
            skin=skin,
            thermostat=thermostat,
        )
        capture_steps = _capture_steps(horizon, stride, max_frames)
        streaming = frame_reducer is not None
        if streaming:
            frame_reducer.begin(
                count=count,
                n_particles=parent.n_particles,
                capture_steps=capture_steps,
                device=parent.device,
                dtype=parent.dtype,
            )
        position_frames: list[torch.Tensor] = []
        unwrapped_frames: list[torch.Tensor] = []
        velocity_frames: list[torch.Tensor] = []
        previous = 0
        for frame_index, target in enumerate(capture_steps):
            if target != previous:
                integrator.step(target - previous)
                previous = target
            positions = _cpu_tensor(system.positions)
            unwrapped = _cpu_tensor(system.unwrapped_positions)
            velocities = _cpu_tensor(system.velocities)
            if streaming:
                # Reduce online and drop the frame -- no (T, B, N, 3) ever forms.
                frame_reducer.update(
                    StreamingFrame(
                        frame_index=frame_index,
                        step=target,
                        positions=positions,
                        unwrapped_positions=unwrapped,
                        velocities=velocities,
                    )
                )
            else:
                position_frames.append(positions)
                unwrapped_frames.append(unwrapped)
                velocity_frames.append(velocities)
        reduction = frame_reducer.result() if streaming else None
        trajectory = (
            None
            if streaming
            else BatchedTrajectory(
                steps=tuple(capture_steps),
                positions=torch.stack(position_frames),
                unwrapped_positions=torch.stack(unwrapped_frames),
                velocities=torch.stack(velocity_frames),
            )
        )
        capture_meta: dict[str, Any] = {
            "mode": "streaming" if streaming else "full",
            "n_frames": len(capture_steps),
            "capture_steps": list(capture_steps),
            # Full capture retains every frame; streaming holds only the current
            # frame (the reducer keeps an O(B*N) accumulator, not a frame list).
            "max_retained_frames": 1 if streaming else len(capture_steps),
            "max_frames": max_frames,
        }
        final_states = tuple(system.branch(index) for index in range(count))
        parent_sha256 = _parent_state_sha256(parent)
        parent_payload: dict[str, Any] = _state_payload(parent)
        parent_payload.update(
            {
                "format_version": 1,
                "state_sha256": parent_sha256,
            }
        )
        run.write_bytes("parent_state.pt", _save_bytes(parent_payload))

        branch_records: list[dict[str, Any]] = []
        for index, final_state in enumerate(final_states):
            label = f"branches/{index:06d}"
            final_state_file = f"{label}/final_state.pt"
            # Streaming skips the ~43 GB per-branch trajectory publication; the
            # online reduction is the retained product, not the frame history.
            trajectory_file = None if streaming else f"{label}/trajectory.pt"
            final_payload: dict[str, Any] = _state_payload(final_state)
            final_payload.update(
                {
                    "format_version": 1,
                    "branch_index": index,
                    "parent_state_sha256": parent_sha256,
                }
            )
            run.write_bytes(
                final_state_file,
                _save_bytes(final_payload),
            )
            if not streaming:
                assert trajectory is not None
                run.write_bytes(
                    trajectory_file,
                    _save_bytes(
                        {
                            "format_version": 1,
                            "branch_index": index,
                            "steps": torch.tensor(trajectory.steps, dtype=torch.int64),
                            "positions": trajectory.positions[:, index].clone(),
                            "unwrapped_positions": trajectory.unwrapped_positions[:, index].clone(),
                            "velocities": trajectory.velocities[:, index].clone(),
                        }
                    ),
                )
            branch_records.append(
                {
                    "index": index,
                    "momentum_seed": int(issued_branch_seeds[index]),
                    "torch_seed": int(branch_torch_seeds[index]),
                    "final_state_file": final_state_file,
                    "trajectory_file": trajectory_file,
                }
            )

        provenance = {
            "format_version": 1,
            "parent_id": parent_id,
            "parent_state_sha256": parent_sha256,
            "parent_state_file": "parent_state.pt",
            "parent": {
                "n_particles": parent.n_particles,
                "dtype": str(parent.dtype),
                "device": str(parent.device),
                "active_particles": int(parent.active_mask.sum().item()),
            },
            "controls": {
                "count": count,
                "temperature": float(temperature),
                "horizon": horizon,
                "dt": float(dt),
                "stride": stride,
                "skin": float(skin),
                "integrator": "velocity_verlet_nve" if thermostat_tau is None else "velocity_verlet_bussi_nvt",
                "thermostat_tau": None if thermostat_tau is None else float(thermostat_tau),
            },
            "capture": {
                "mode": capture_meta["mode"],
                "n_frames": capture_meta["n_frames"],
                "max_frames": capture_meta["max_frames"],
                "max_retained_frames": capture_meta["max_retained_frames"],
                "reducer": None if not streaming else type(frame_reducer).__name__,
            },
            "trajectory_steps": list(capture_steps),
            "momentum_seed_domain": momentum_seed_domain,
            "thermostat": None
            if thermostat_seed is None
            else {
                "seed_domain": thermostat_seed_domain,
                "issued_seed": int(thermostat_seed),
                "torch_seed": int(thermostat_torch_seed),
                "stream_layout": "step-major, then branch-major contiguous ndof normal blocks",
            },
            "branches": branch_records,
        }
        run.write_json("branch_provenance.json", provenance)
        run.log(
            f"published {count} branching {capture_meta['mode']} runs through step {horizon} "
            f"({len(capture_steps)} frames, {provenance['controls']['integrator']})"
        )
        run.finish("completed")
    except BaseException:
        run.log("branch ensemble failed before complete publication")
        try:
            run.finish("failed")
        except Exception:
            # The original error is more useful, and a caller could have
            # explicitly finalized a run before passing it here.
            pass
        raise

    return BranchEnsembleResult(
        run=run,
        branch_seeds=issued_branch_seeds,
        branch_torch_seeds=branch_torch_seeds,
        thermostat_seed=thermostat_seed,
        thermostat_torch_seed=thermostat_torch_seed,
        final_states=final_states,
        trajectory=trajectory,
        reduction=reduction,
        capture_meta=capture_meta,
    )
