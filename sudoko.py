from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a 4-qubit, 4-classical bit circuit
qc = QuantumCircuit(4, 4)

#Initialize in superposition
qc.h([0, 1, 2, 3])


# Flip for 0110
qc.x([0, 3])         # Flip bits so 0110 becomes 0000
qc.h(3)
qc.mct([0, 1, 2], 3) # If q0,q1,q2 are 0, mark q3
qc.h(3)
qc.x([0, 3])         # Unflip

# Flip for 1001
qc.x([1, 2])
qc.h(3)
qc.mct([0, 1, 2], 3)
qc.h(3)
qc.x([1, 2])

# Step 3: Grover diffusion
qc.h([0, 1, 2, 3])
qc.x([0, 1, 2, 3])
qc.h(3)
qc.mct([0, 1, 2], 3)
qc.h(3)
qc.x([0, 1, 2, 3])
qc.h([0, 1, 2, 3])

# Measure
qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

#  Run
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
counts = result.get_counts()

# Display
print("Results:", counts)
plot_histogram(counts)
plt.show()
