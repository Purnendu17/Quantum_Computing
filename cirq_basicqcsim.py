import cirq
import numpy as np
import matplotlib.pyplot as plt
from math import log2

np.set_printoptions(precision=6, suppress=True)

# --- Utilities ---
def idx_to_basis(n_qubits, idx):
    return format(idx, '0{}b'.format(n_qubits))

# --- Partial trace ---
def partial_trace(rho, keep, n_qubits):
    keep = list(keep)
    dim_keep = 2**len(keep)
    result = np.zeros((dim_keep, dim_keep), dtype=complex)
    for i in range(2**n_qubits):
        for j in range(2**n_qubits):
            bi = list(map(int, format(i, f'0{n_qubits}b')))
            bj = list(map(int, format(j, f'0{n_qubits}b')))
            if all(bi[q] == bj[q] for q in range(n_qubits) if q not in keep):
                idx_i = idx_j = 0
                for q in keep:
                    idx_i = (idx_i << 1) | bi[q]
                    idx_j = (idx_j << 1) | bj[q]
                result[idx_i, idx_j] += rho[i, j]
    return result

def concurrence_from_statevector(psi):
    a, b, c, d = psi
    return 2 * abs(a * d - b * c)

# --- Circuits ---
def make_bell_circuit(which="Phi+"):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))
    circuit.append(cirq.CNOT(q0, q1))

    if which == "Phi-":
        circuit.append(cirq.Z(q0))
    elif which == "Psi+":
        circuit.append(cirq.X(q1))
    elif which == "Psi-":
        circuit.append([cirq.Z(q0), cirq.X(q1)])

    return circuit, (q0, q1)

# --- Analysis ---
def analyze_state(circuit, qubits, name="state"):
    sim = cirq.Simulator()

    # --- simulate to get statevector ---
    sim_result = sim.simulate(circuit)

    # Cirq version differences
    if hasattr(sim_result, "final_state_vector"):
        psi = sim_result.final_state_vector  # older cirq
    else:
        psi = sim_result.final_state  # newer cirq

    # density matrix
    rho = np.outer(psi, np.conjugate(psi))

    # partial traces
    n = 2
    rhoA = partial_trace(rho, [0], n)
    rhoB = partial_trace(rho, [1], n)

    # von Neumann entropy
    def von_neumann_entropy(rho_mat):
        vals = np.linalg.eigvals(rho_mat)
        vals = np.real_if_close(vals)
        return -sum(v * log2(v) for v in vals if v > 1e-12)

    entropyA = von_neumann_entropy(rhoA)
    entropyB = von_neumann_entropy(rhoB)
    concurrence = concurrence_from_statevector(psi)

    print(f"\n=== {name} ===")
    print("Circuit:\n", circuit)
    print("Statevector:", psi)
    print("Density matrix:\n", rho)
    print("ρ_A:\n", rhoA)
    print("ρ_B:\n", rhoB)
    print(f"Entropy(ρ_A)={entropyA:.6f}, Entropy(ρ_B)={entropyB:.6f}")
    print(f"Concurrence={concurrence:.6f}")
    print("Reduced states maximally mixed?",
          np.allclose(rhoA, np.eye(2)/2, atol=1e-8),
          np.allclose(rhoB, np.eye(2)/2, atol=1e-8))

    # --- measurement ---
    circuit_meas = circuit.copy()
    circuit_meas.append(cirq.measure(*qubits, key="m"))
    run_res = sim.run(circuit_meas, repetitions=1024)
    counts = run_res.histogram(key="m")

    labels_sorted = [format(k, f'0{len(qubits)}b') for k in sorted(counts.keys())]
    values_sorted = [counts[k] for k in sorted(counts.keys())]

    plt.bar(labels_sorted, values_sorted)
    plt.title(f"{name} measurement outcomes")
    plt.show()
    print("Counts:", dict(zip(labels_sorted, values_sorted)))


# --- Run all 4 Bell states ---
for bell in ["Phi+", "Phi-", "Psi+", "Psi-"]:
    circuit, qubits = make_bell_circuit(bell)
    analyze_state(circuit, qubits, bell)
