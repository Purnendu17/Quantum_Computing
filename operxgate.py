import numpy as np
import matplotlib.pyplot as plt

# Define basis states
zero = np.array([[1], [0]], dtype=complex)  # |0>
one = np.array([[0], [1]], dtype=complex)   # |1>

# Define Pauli-X gate
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

# ---- Choose your input state ----
a, b = 1/np.sqrt(2), 1/np.sqrt(2)   # equal superposition
state = a*zero + b*one

print("Initial state vector =")
print(state)

# Apply X gate
new_state = X @ state

print("\nAfter applying X gate, new state =")
print(new_state)

# Compute probabilities
probs = np.abs(new_state.flatten())**2
labels = ["|0>", "|1>"]

# Plot probability distribution
plt.bar(labels, probs, color="green")
plt.title("Probabilities after X gate")
plt.ylabel("Probability")
plt.ylim(0,1)
plt.show()
