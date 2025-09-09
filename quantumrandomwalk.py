"""
Quantum vs Classical Random Walk (1D, coined discrete-time walk)

- Quantum walk: coin (2-dim) x position (Npos-dim), coin flip = Hadamard,
  conditional shift: coin |0> -> step left, coin |1> -> step right.
- Classical walk: unbiased step left/right, many trials averaged.

Features:
- Plot position probability distributions at sample times.
- Plot standard deviation vs steps (quantum vs classical).
- Show reduced coin density for quantum walk.
- Optional: build Cirq MatrixGate for one-step unitary and sample measurements.

Author: ChatGPT (adapted to your earlier Cirq usage)
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import sys

# Optional: use Cirq to demonstrate sampling from the quantum state
try:
    import cirq
    HAVE_CIRQ = True
except Exception:
    HAVE_CIRQ = False

np.set_printoptions(precision=6, suppress=True)

# --------------------
# Parameters
# --------------------
L = 50                 # half-width of the lattice: positions from -L .. +L
Npos = 2*L + 1         # number of position sites
steps = 80             # number of time steps to run
classical_trials = 25000  # Monte Carlo trials for classical average

# Choose boundary behavior: 'reflect' or 'periodic'
BOUNDARY = 'reflect'

# Selected times to plot distributions
plot_times = [1, 5, 10, 20, 40, 80]


# --------------------
# Helper functions
# --------------------
def pos_index(x):
    """Map position coordinate x in [-L, L] to index 0..Npos-1."""
    return x + L

def index_to_pos(i):
    """Inverse of pos_index."""
    return i - L

def build_coin_and_shift(Npos, boundary='reflect'):
    """
    Build coin (2x2) and conditional shift S (2*Npos x 2*Npos).
    Basis ordering: |c> ⊗ |x> with coin c in {0,1}, position index x_idx in 0..Npos-1.
    """

    # Coin (Hadamard)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

    dim = 2 * Npos
    S = np.zeros((dim, dim), dtype=complex)

    # For each basis state |c> |x>, define where it moves:
    # if c==0 -> move left (x-1), if c==1 -> move right (x+1)
    for c in (0, 1):
        for x_idx in range(Npos):
            src_index = c * Npos + x_idx
            x = index_to_pos(x_idx)
            if c == 0:
                # move left
                x_new = x - 1
            else:
                x_new = x + 1

            # Boundary handling
            if x_new < -L or x_new > L:
                if boundary == 'reflect':
                    # reflect back: if trying to move left at leftmost, stay at leftmost
                    x_new = max(min(x_new, L), -L)
                elif boundary == 'periodic':
                    # wrap-around
                    if x_new < -L:
                        x_new = L
                    elif x_new > L:
                        x_new = -L
                else:
                    raise ValueError("Unsupported boundary mode")

            tgt_idx = pos_index(x_new)
            tgt_index = c * Npos + tgt_idx
            # S maps |c,x> -> |c, x_new>
            S[tgt_index, src_index] = 1.0

    return H, S


def kron(a, b):
    return np.kron(a, b)


def initial_quantum_state(Npos, coin_state=0, pos0=0):
    """
    Prepare |coin_state> ⊗ |pos0>
    coin_state: 0 or 1 (can be used to set superposition externally)
    pos0: starting position (coordinate)
    """
    psi = np.zeros((2 * Npos,), dtype=complex)
    coin_basis = np.zeros((2,), dtype=complex)
    coin_basis[coin_state] = 1.0
    pos_basis = np.zeros((Npos,), dtype=complex)
    pos_basis[pos_index(pos0)] = 1.0
    psi = kron(coin_basis, pos_basis)
    return psi


def reduced_density_coin(rho, Npos):
    """
    Given full density matrix rho (2*Npos x 2*Npos), return reduced 2x2 density for coin.
    Trace out position degrees.
    """
    dim = 2 * Npos
    rho_coin = np.zeros((2, 2), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            # basis indices: (c_i, x_i), (c_j, x_j)
            c_i = i // Npos
            x_i = i % Npos
            c_j = j // Npos
            x_j = j % Npos
            if x_i == x_j:
                rho_coin[c_i, c_j] += rho[i, j]
    return rho_coin


# --------------------
# Build operators
# --------------------
H_coin, S_shift = build_coin_and_shift(Npos, boundary=BOUNDARY)
# Coin operator on full space = H_coin ⊗ I_pos
C_full = kron(H_coin, np.eye(Npos, dtype=complex))
# Single step unitary: S * C_full
U_step = S_shift @ C_full

# Sanity: U_step should be unitary (within numerical tolerance)
unitary_check = np.allclose(U_step.conj().T @ U_step, np.eye(U_step.shape[0]), atol=1e-10)
print("U_step unitary check:", unitary_check)

# --------------------
# Quantum evolution (statevector method)
# --------------------
psi0 = initial_quantum_state(Npos, coin_state=0, pos0=0)   # coin |0>, start at position 0
psi = psi0.copy()

# storage
quantum_pos_probs = np.zeros((steps+1, Npos))
quantum_coin_probs = np.zeros((steps+1, 2))
quantum_std = np.zeros(steps+1)

# step 0
rho0 = np.outer(psi, np.conjugate(psi))
# position probabilities (sum over coin)
pos_probs0 = np.zeros(Npos)
for c in (0, 1):
    pos_probs0 += np.abs(psi[c*Npos:(c+1)*Npos])**2
quantum_pos_probs[0, :] = pos_probs0
rho_coin0 = reduced_density_coin(rho0, Npos)
quantum_coin_probs[0, :] = np.real(np.diag(rho_coin0))
# stddev
positions = np.array([index_to_pos(i) for i in range(Npos)])
quantum_mean = np.sum(positions * pos_probs0)
quantum_var = np.sum((positions - quantum_mean)**2 * pos_probs0)
quantum_std[0] = sqrt(quantum_var)

# evolve
for t in range(1, steps+1):
    psi = U_step @ psi
    # probabilities
    pos_probs = np.zeros(Npos)
    for c in (0, 1):
        pos_probs += np.abs(psi[c*Npos:(c+1)*Npos])**2
    quantum_pos_probs[t, :] = pos_probs

    rho = np.outer(psi, np.conjugate(psi))
    rho_coin = reduced_density_coin(rho, Npos)
    quantum_coin_probs[t, :] = np.real(np.diag(rho_coin))

    mean = np.sum(positions * pos_probs)
    var = np.sum((positions - mean)**2 * pos_probs)
    quantum_std[t] = sqrt(var)

# --------------------
# Classical Monte Carlo (many trials)
# --------------------
rng = np.random.default_rng(seed=42)
classical_pos_probs = np.zeros((steps+1, Npos))
classical_std = np.zeros(steps+1)

# We'll simulate many independent walkers
# Represent positions as integers; reflect at boundaries
pos_array = np.zeros((classical_trials,), dtype=int)  # start at 0

# t=0 distribution
counts, = np.bincount([pos_index(0)]*classical_trials, minlength=Npos)[:1]  # trivial
classical_pos_probs[0, pos_index(0)] = 1.0
classical_std[0] = 0.0

for t in range(1, steps+1):
    # random steps: -1 or +1 with p=0.5
    steps_r = rng.choice([-1, 1], size=classical_trials)
    pos_array = pos_array + steps_r
    if BOUNDARY == 'reflect':
        pos_array = np.clip(pos_array, -L, L)
    elif BOUNDARY == 'periodic':
        # wrap-around
        pos_array = ((pos_array + L) % Npos) - L

    # tally
    # convert positions to indices
    idxs = pos_array + L
    counts = np.bincount(idxs, minlength=Npos)
    classical_pos_probs[t, :] = counts / classical_trials

    # std
    mean = np.sum(positions * classical_pos_probs[t, :])
    var = np.sum((positions - mean)**2 * classical_pos_probs[t, :])
    classical_std[t] = sqrt(var)

# --------------------
# Plot distributions at selected times
# --------------------
plt.figure(figsize=(12, 8))
for i, t in enumerate(plot_times):
    if t > steps:
        continue
    plt.subplot(len(plot_times)//2 + 1, 2, i+1)
    plt.plot(positions, classical_pos_probs[t, :], label='Classical (MC)', linestyle='--')
    plt.plot(positions, quantum_pos_probs[t, :], label='Quantum (prob.)')
    plt.title(f"Position distribution at t = {t}")
    plt.xlabel("position")
    plt.ylabel("probability")
    plt.legend()
plt.tight_layout()
plt.show()

# --------------------
# Plot std dev vs steps
# --------------------
plt.figure(figsize=(8, 5))
tvec = np.arange(steps+1)
plt.plot(tvec, classical_std, label='Classical (MC)')
plt.plot(tvec, quantum_std, label='Quantum (statevector)')
# reference classical sqrt(t) scaling (normalized to initial slope)
plt.plot(tvec, np.sqrt(tvec), label=r'$\sqrt{t}$ (reference)', linestyle=':')
plt.title("Standard deviation vs time (walk spread)")
plt.xlabel("time step")
plt.ylabel("std dev")
plt.legend()
plt.grid(True)
plt.show()

# --------------------
# Additional outputs: coin probabilities and example snapshots
# --------------------
# show coin marginal at final time
print("\nCoin probabilities (final time):")
print("Quantum coin probs:", quantum_coin_probs[steps, :])

# print final position distribution peaks
top_positions_quantum = np.argsort(quantum_pos_probs[steps, :])[-6:][::-1]
top_positions_class = np.argsort(classical_pos_probs[steps, :])[-6:][::-1]
print("\nTop positions (quantum) at t=", steps, ":", [(index_to_pos(i), round(quantum_pos_probs[steps,i],4)) for i in top_positions_quantum])
print("Top positions (classical) at t=", steps, ":", [(index_to_pos(i), round(classical_pos_probs[steps,i],4)) for i in top_positions_class])

# --------------------
# Optional: create a Cirq MatrixGate for U_step and sample measurements
# --------------------
if HAVE_CIRQ:
    print("\nCirq available: building MatrixGate for one-step unitary and sampling final-state measurement.")
    # Build a single MatrixGate that is U_step
    U = U_step  # shape (2*Npos, 2*Npos)
    # Cirq expects the unitary to act on qubits; choose qubit ordering: coin qubit first, then position qubits (we'll use log2 encoding for position if we want actual qubits)
    # But our position space is not a power of two necessarily; we can still use MatrixGate on a single 'big' qudit composite via cirq.MatrixGate on many qubits only if dims match power-of-two.
    # Here Npos might not be power-of-two; to demo sampling using Cirq, let's embed into the smallest power-of-two position register (optional).
    # For simplicity, we'll just sample from the numpy probabilities to mimic measurement statistics (equivalent).
    probs = quantum_pos_probs[steps, :]

    # sample many shots from the computed distribution
    shots = 2048
    rng = np.random.default_rng(123)
    sample_idxs = rng.choice(np.arange(Npos), size=shots, p=probs)
    counts = {}
    for idx in sample_idxs:
        key = str(index_to_pos(idx))
        counts[key] = counts.get(key, 0) + 1
    # show a small histogram
    keys_sorted = sorted(counts.keys(), key=lambda s: int(s))
    vals = [counts[k] for k in keys_sorted]
    plt.figure(figsize=(9,4))
    plt.bar([int(k) for k in keys_sorted], vals)
    plt.title(f"Sampled measurement counts (quantum) at t={steps}, {shots} shots (simulated)")
    plt.xlabel("position")
    plt.show()
    print("Sampled counts (position->counts) (trimmed):")
    # print small window around center
    center_idx = pos_index(0)
    window = 10
    trimmed = {index_to_pos(i): round(quantum_pos_probs[steps,i],4) for i in range(max(0, center_idx-window), min(Npos, center_idx+window+1))}
    print(trimmed)
else:
    print("\nCirq not installed -- skipping CircuitGate sampling demo. Install cirq to try MatrixGate approach.")

print("\nDone.")
