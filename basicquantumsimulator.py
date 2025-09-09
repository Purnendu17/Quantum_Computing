import numpy as np
import matplotlib.pyplot as plt
from math import log2, isclose

np.set_printoptions(precision=6, suppress=True)

# --- Utilities ---
def kron(a, b): return np.kron(a, b)

def tensor_list(matrices):
    """Tensor a list of matrices in order: leftmost = most-significant qubit."""
    result = matrices[0]
    for m in matrices[1:]:
        result = kron(result, m)
    return result

def idx_to_basis(n_qubits, idx):
    """Return basis label string (e.g. 2 -> '10' for n_qubits=2)."""
    return format(idx, '0{}b'.format(n_qubits))

# --- Gate definitions (dictionary style) ---
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

single_qubit_gates = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z,
    'H': H
}

# --- State preparation ---
def zero_state(n_qubits):
    dim = 2**n_qubits
    psi = np.zeros((dim, 1), dtype=complex)
    psi[0, 0] = 1.0
    return psi

# --- Growing gates ---
def grow_single_to_full(n_qubits, gate, target):
    ops = [gate if q==target else I for q in range(n_qubits)]
    return tensor_list(ops)

def cnot_full(n_qubits, ctrl, tgt):
    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(map(int, format(i, f'0{n_qubits}b')))
        if bits[ctrl] == 1:
            bits_flipped = bits.copy()
            bits_flipped[tgt] ^= 1
            j = int(''.join(map(str, bits_flipped)), 2)
        else:
            j = i
        full[j, i] = 1.0
    return full

# --- Density matrix & partial trace ---
def density_matrix(psi): return psi @ psi.conj().T

def partial_trace(rho, keep, n_qubits):
    keep = list(keep)
    dim_keep = 2**len(keep)
    result = np.zeros((dim_keep, dim_keep), dtype=complex)
    for i in range(2**n_qubits):
        for j in range(2**n_qubits):
            bi = list(map(int, format(i, f'0{n_qubits}b')))
            bj = list(map(int, format(j, f'0{n_qubits}b')))
            if all(bi[q]==bj[q] for q in range(n_qubits) if q not in keep):
                idx_i = idx_j = 0
                for q in keep:
                    idx_i = (idx_i<<1) | bi[q]
                    idx_j = (idx_j<<1) | bj[q]
                result[idx_i, idx_j] += rho[i,j]
    return result

# --- Entropy & Concurrence ---
def von_neumann_entropy(rho):
    vals = np.real_if_close(np.linalg.eigvals(rho))
    return -sum(v*log2(v) for v in vals if v>1e-12)

def concurrence_from_statevector(psi):
    vec = psi.flatten()
    a,b,c,d = vec
    return 2*abs(a*d - b*c)

# --- Bell state generator ---
def make_bell(which="Phi+"):
    n=2
    psi = zero_state(n)
    H0 = grow_single_to_full(n,H,0)
    CNOT = cnot_full(n,0,1)
    psi = CNOT @ (H0 @ psi)  # base |Φ+>
    if which=="Phi-":
        psi = grow_single_to_full(n,Z,0) @ psi
    elif which=="Psi+":
        psi = grow_single_to_full(n,X,1) @ psi
    elif which=="Psi-":
        psi = grow_single_to_full(n,Z,0) @ (grow_single_to_full(n,X,1) @ psi)
    return psi/np.linalg.norm(psi)

# --- Verification and measurement ---
def analyze_state(psi, name="state"):
    n=2
    rho = density_matrix(psi)
    rhoA = partial_trace(rho,[0],n)
    rhoB = partial_trace(rho,[1],n)
    entropyA = von_neumann_entropy(rhoA)
    entropyB = von_neumann_entropy(rhoB)
    concur = concurrence_from_statevector(psi)

    print(f"\n=== {name} ===")
    print("Statevector:", psi.flatten())
    print("Density matrix:\n", rho)
    print("ρ_A:\n", rhoA)
    print("ρ_B:\n", rhoB)
    print(f"Entropy(ρ_A)={entropyA:.6f}, Entropy(ρ_B)={entropyB:.6f}")
    print(f"Concurrence={concur:.6f}")
    print("Reduced states maximally mixed?",
          np.allclose(rhoA, np.eye(2)/2, atol=1e-8),
          np.allclose(rhoB, np.eye(2)/2, atol=1e-8))

    # measurement
    probs = (np.abs(psi.flatten())**2).real
    shots = 1024
    outcomes = np.random.choice(2**n, size=shots, p=probs)
    counts = {idx_to_basis(n,k):0 for k in range(2**n)}
    for o in outcomes: counts[idx_to_basis(n,o)]+=1
    plt.bar(counts.keys(), counts.values())
    plt.title(f"{name} measurement outcomes")
    plt.show()
    print("Counts:", counts)

# --- Run all 4 Bell states ---
for bell in ["Phi+","Phi-","Psi+","Psi-"]:
    psi = make_bell(bell)
    analyze_state(psi, bell)
