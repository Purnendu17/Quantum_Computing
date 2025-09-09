import numpy as np
import matplotlib.pyplot as plt

# Define basic states
zero = np.array([[1], [0]], dtype=complex)
one = np.array([[0], [1]], dtype=complex)

# H gate
H = (1/np.sqrt(2)) * np.array([[1, 1],
                                [1, -1]], dtype=complex)

# 2-qubit H 
H2 = np.kron(H, H)

# Identity (2 qubits)
I4 = np.eye(4)

# Initial state |00>
init_state = np.kron(zero, zero)

# Apply HXH to get superposition
state = H2 @ init_state

# Define Oracle
oracle = np.eye(4)
oracle[3, 3] = -1  # flip phase of |11>

# Diffusion operator
s = (1/2) * np.ones((4,1))   # uniform superposition
D = 2 * (s @ s.T) - I4

# One Grover iteration
state = oracle @ state
state = D @ state

# Compute probabilities
probs = np.abs(state.flatten())**2

# Plot result
labels = ["|00>", "|01>", "|10>", "|11>"]
plt.bar(labels, probs, color="purple")
plt.title("Grover's Algorithm (2 qubits, marked state = |11>)")
plt.ylabel("Probability")
plt.ylim(0,1)
plt.show()

print("Final state amplitudes:", state.flatten())
print("Measurement probabilities:", probs)
