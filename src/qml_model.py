from __future__ import annotations

import math
from dataclasses import dataclass
from math import pi
from typing import List, Tuple

import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit import ParameterVector
# Qiskit Aer renamed Sampler implementations across releases. Prefer the new
# SamplerV2, but fall back to the legacy Sampler when running with the older
# pinned stack to keep installs lightweight.
try:  # pragma: no cover - import shim
    from qiskit_aer.primitives import SamplerV2 as AerSampler
except ImportError:  # pragma: no cover - legacy fallback
    from qiskit_aer.primitives import Sampler as AerSampler

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from .qml_layers import BitstringSelectLayer, NormLayer, ReshapeSumLayer


@dataclass(frozen=True)
class QNNSettings:
    n_outputs: int = 2
    reps: int = 2
    norm: bool = True
    shots: int = 10_000
    seed: int = 42
    padding_value: float = 2.5 * pi
    readout: str = "sum"
    bitstrings: tuple[str, ...] | None = None

    # Optional compilation/transpilation
    transpile: bool = False
    optimization_level: int = 1
    seed_transpiler: int | None = None


def _build_parametrized_vqc(
    n_qubits: int,
    reps: int,
) -> Tuple[QuantumCircuit, List, List]:
    """Build a simple ansatz with separated input vs weight parameters.

    This mirrors our current ansatz conceptually (encoding RY + trainable RY/RZ + CX chain),
    but expressed in the CircuitQNN style: input_params vs weight_params.

    Returns: (qc, input_params, weight_params)
    """

    x_params = ParameterVector("x", n_qubits)
    theta_params = ParameterVector("theta", reps * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)

    # Encoding: RY(x_i) on each qubit
    for i in range(n_qubits):
        qc.ry(x_params[i], i)

    # Variational layers
    idx = 0
    for _ in range(reps):
        for i in range(n_qubits):
            qc.ry(theta_params[idx], i)
            idx += 1
            qc.rz(theta_params[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        # Sampler-based QNNs operate on measurement distributions, so the circuit
        # must include measurements.
    qc.measure_all()

    input_params = list(x_params)
    weight_params = list(theta_params)
    return qc, input_params, weight_params


def _default_bitstrings(n_outputs: int, n_qubits: int) -> tuple[str, ...]:
    if n_outputs > 2**n_qubits:
        raise ValueError("Requested more outputs than available basis states.")
    if n_outputs == 1:
        return ("0" * n_qubits,)
    if n_outputs == 2:
        return ("0" * n_qubits, "1" * n_qubits)
    width = n_qubits
    return tuple(format(i, f"0{width}b") for i in range(n_outputs))


def _resolve_bitstrings(
    settings: QNNSettings,
    *,
    n_qubits: int,
    n_outputs: int,
) -> tuple[str, ...]:
    if settings.bitstrings is not None:
        bitstrings = tuple(settings.bitstrings)
    else:
        bitstrings = _default_bitstrings(n_outputs=n_outputs, n_qubits=n_qubits)

    if len(bitstrings) != n_outputs:
        raise ValueError("Number of bitstrings must equal n_outputs when using bitstring readout")

    widths = {len(bs) for bs in bitstrings}
    if widths != {n_qubits}:
        raise ValueError(
            f"Bitstrings must match the circuit width ({n_qubits} qubits); got widths={sorted(widths)}"
        )
    return bitstrings


def build_qnn_base_model(
    n_inputs: int,
    settings: QNNSettings = QNNSettings(),
) -> torch.nn.Module:
    """Build the *base* QNN model up through the pre-interpretation readout.

    The interpretation layer (linear/rational/rationalLog) is applied separately.
    """

    n_outputs = int(settings.n_outputs)
    if n_outputs <= 0:
        raise ValueError("n_outputs must be positive")

    # Ensure enough qubits to represent at least `n_outputs` distinct basis states.
    n_qubits_out = math.ceil(math.log2(n_outputs))
    # Also ensure at least one input angle per qubit. If we end up with more
    # qubits than inputs, we pad the input vector below.
    n_qubits = max(int(n_inputs), int(n_qubits_out))

    qc, input_params, weight_params = _build_parametrized_vqc(n_qubits=n_qubits, reps=int(settings.reps))

    if settings.transpile:
        # Optional: transpilation can change the circuit structure. Keep
        # `seed_transpiler` explicit for reproducibility when enabled.
        qc = transpile(
            qc,
            optimization_level=int(settings.optimization_level),
            seed_transpiler=int(settings.seed_transpiler) if settings.seed_transpiler is not None else None,
        )

    # Qiskit Aer primitive APIs/options vary across versions; this shim supports
    # both the pinned stack and newer installs.
    try:
        sampler = AerSampler(default_shots=int(settings.shots), seed=int(settings.seed))
    except TypeError:
        sampler = AerSampler()
        set_options = getattr(sampler, "set_options", None)
        if callable(set_options):
            set_options(shots=int(settings.shots), seed_simulator=int(settings.seed))
        elif hasattr(sampler, "options"):
            sampler.options.shots = int(settings.shots)

    # Identity interpretation: the QNN returns the full probability vector over
    # basis states. Downstream readout layers reduce this to `n_outputs`.
    output_shape = 2**n_qubits
    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=input_params,
        weight_params=weight_params,
        interpret=lambda i: i,
        output_shape=output_shape,
    )

    # QCardEst-style initial weights: uniform in [-pi, pi].
    rng = np.random.default_rng(int(settings.seed))
    initial_weights = (2 * pi * rng.random(qnn.num_weights) - pi)

    quantum_layer = TorchConnector(qnn, initial_weights)

    model = torch.nn.Sequential()

    # If the circuit expects `n_qubits` inputs but we only provide `n_inputs`
    # angles, pad the remainder with a constant (mirrors upstream practice).
    if n_qubits > n_inputs:
        model.append(torch.nn.ConstantPad1d((0, n_qubits - n_inputs), float(settings.padding_value)))

    model.append(quantum_layer)

    readout = str(settings.readout).lower()
    if readout == "sum":
        # Legacy/QCardEst-style reduction: reshape into (n_outputs, -1) and sum
        # each bin.
        model.append(ReshapeSumLayer(n_outputs))
    elif readout == "bitstring":
        # Explicit reduction: select K basis-state indices by bitstring.
        bitstrings = _resolve_bitstrings(settings, n_qubits=n_qubits, n_outputs=n_outputs)
        model.append(BitstringSelectLayer(bitstrings))
    else:
        raise ValueError(f"Unknown readout strategy: {settings.readout}")

    if settings.norm:
        # Max-normalization (x / x.max()) like QCardEst; not sum-to-1.
        model.append(NormLayer())

    return model
