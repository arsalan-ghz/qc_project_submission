from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class AngleMap:
    clip: float = 2.0  # clip z-scores to [-clip, clip]
    scale: float = np.pi / 2  # map to [-pi, pi] when clip=2


def angles_from_z(z: np.ndarray, angle_map: AngleMap = AngleMap()) -> np.ndarray:
    """Map standardized features z to angles.

    Used by `run_qcardest_style_training.py` to turn z-scored features into
    rotation angles for the QNN input parameters.

    Design:
    - clip z-scores to [-clip, clip] to avoid extreme rotations
    - scale into an angle range (default scale maps clip=2 to roughly [-pi, pi])
    """
    z = np.clip(z, -angle_map.clip, angle_map.clip)
    return z * angle_map.scale


def build_vqc_circuit(
    x_angles: np.ndarray,
    theta: np.ndarray,
    n_qubits: int = 3,
    reps: int = 1,
    with_measure: bool = False,
) -> QuantumCircuit:
    """
    Minimal VQC:
      - Angle encoding with RY(x_i) on qubit i
      - Trainable layer: RY, RZ per qubit
      - Entangling chain CX(0,1), CX(1,2), ...
      - reps = 1 (or 2)
    theta shape: (reps, n_qubits, 2) -> [RY, RZ]
    """
    qc = QuantumCircuit(n_qubits)

    # Encoding
    for i in range(n_qubits):
        qc.ry(float(x_angles[i]), i)

    # Variational layers
    for r in range(reps):
        for i in range(n_qubits):
            qc.ry(float(theta[r, i, 0]), i)
            qc.rz(float(theta[r, i, 1]), i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    if with_measure:
        qc.measure_all()

    return qc


def probs_from_statevector(qc: QuantumCircuit) -> Dict[str, float]:
    """Exact probabilities from statevector (no shots)."""
    # Ensure no measurement in circuit passed here.
    sv = Statevector.from_instruction(qc)
    return sv.probabilities_dict()


def delta_hat_linear(probs: Dict[str, float], a: float, b: float, n_qubits: int) -> float:
    """Linear architecture: Δ = a * p(00..0) + b"""
    key0 = "0" * n_qubits
    p0 = float(probs.get(key0, 0.0))
    return a * p0 + b

def delta_hat_rational(probs: Dict[str, float], a: float, b: float, n_qubits: int, eps: float = 1e-6) -> float:
    """Rational architecture: Δ = a * p(00..0) / (p(11..1) + eps) + b"""
    key0 = "0" * n_qubits
    key1 = "1" * n_qubits
    p0 = float(probs.get(key0, 0.0))
    p1 = float(probs.get(key1, 0.0))
    return a * (p0 / (p1 + eps)) + b


def delta_hat_rationallog(probs: Dict[str, float], a: float, b: float, n_qubits: int, eps: float = 1e-6) -> float:
    """RationalLog architecture: Δ = a * log((p0+eps)/(p1+eps)) + b"""
    key0 = "0" * n_qubits
    key1 = "1" * n_qubits
    p0 = float(probs.get(key0, 0.0))
    p1 = float(probs.get(key1, 0.0))
    return a * np.log((p0 + eps) / (p1 + eps)) + b


def pack_params(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.concatenate([theta.ravel(), np.array([a, b], dtype=float)])


def unpack_params(p: np.ndarray, reps: int, n_qubits: int) -> Tuple[np.ndarray, float, float]:
    n_theta = reps * n_qubits * 2
    theta = p[:n_theta].reshape((reps, n_qubits, 2))
    a, b = float(p[n_theta]), float(p[n_theta + 1])
    return theta, a, b


def predict_deltas_statevector(
    X_angles: np.ndarray,
    p: np.ndarray,
    reps: int,
    n_qubits: int,
    arch: str = "linear",
    eps: float = 1e-6,
) -> np.ndarray:
    theta, a, b = unpack_params(p, reps=reps, n_qubits=n_qubits)
    out = np.zeros(len(X_angles), dtype=float)
    for i, xang in enumerate(X_angles):
        qc = build_vqc_circuit(xang, theta, n_qubits=n_qubits, reps=reps, with_measure=False)
        probs = probs_from_statevector(qc)
        # --- architecture mapping (traceable) ---
        # out[i] = delta_hat_linear(probs, a=a, b=b, n_qubits=n_qubits)  # (OLD) Linear only

        # (NEW) Select architecture:
        if arch == "linear":
            out[i] = delta_hat_linear(probs, a=a, b=b, n_qubits=n_qubits)
        elif arch == "rational":
            out[i] = delta_hat_rational(probs, a=a, b=b, n_qubits=n_qubits, eps=eps)
        elif arch == "rationallog":
            out[i] = delta_hat_rationallog(probs, a=a, b=b, n_qubits=n_qubits, eps=eps)
        else:
            raise ValueError(f"Unknown arch: {arch}")
    return out


def mse_loss_statevector(
    p: np.ndarray,
    X_angles: np.ndarray,
    y_delta: np.ndarray,
    reps: int,
    n_qubits: int,
    arch: str = "linear",
    eps: float = 1e-6,
) -> float:
    pred = predict_deltas_statevector(X_angles, p, reps=reps, n_qubits=n_qubits, arch=arch, eps=eps)
    return float(np.mean((pred - y_delta) ** 2))
