import numpy as np
import matplotlib.pyplot as plt

# Define basis states |0> and |1>
zero = np.array([[1], [0]], dtype=complex)
one = np.array([[0], [1]], dtype=complex)

# Define Hadamard gate
H = (1/np.sqrt(2)) * np.array([[1, 1],
                                [1, -1]], dtype=complex)

# Apply H to |0> and |1>
state0 = H @ zero   # H|0>
state1 = H @ one    # H|1>

# Compute probabilities
probs0 = [np.abs(state0[0,0])**2, np.abs(state0[1,0])**2]
probs1 = [np.abs(state1[0,0])**2, np.abs(state1[1,0])**2]

print("\nProbabilities for H|0>:")
print("|0>:", np.abs(state0[0,0])**2, " |1>:", np.abs(state0[1,0])**2)

print("\nProbabilities for H|1>:")
print("|0>:", np.abs(state1[0,0])**2, " |1>:", np.abs(state1[1,0])**2)

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(10,4))

# For H|0>
axs[0].bar(["|0>", "|1>"], probs0, color=["blue", "orange"])
axs[0].set_title("Measurement probabilities for H|0>")
axs[0].set_ylim(0,1)
axs[0].set_ylabel("Probability")

# For H|1>
axs[1].bar(["|0>", "|1>"], probs1, color=["blue", "orange"])
axs[1].set_title("Measurement probabilities for H|1>")
axs[1].set_ylim(0,1)

plt.tight_layout()
plt.show()
