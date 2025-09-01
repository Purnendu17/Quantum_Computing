from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Ask for initial state
initial_state = input("Enter initial state (0 or 1): ").strip()

# Build circuit
qc = QuantumCircuit(1, 1)
if initial_state == '1':
    qc.x(0)
elif initial_state != '0':
    print("Invalid input. Defaulting to |0⟩.")

qc.h(0)
qc.h(0)
qc.measure(0, 0)

# Get simulator backend
simulator = Aer.get_backend('qasm_simulator')

# Submit the job directly to the backend
job = simulator.run(qc, shots=1024)

# Get the results
result = job.result()
counts = result.get_counts()

# Show result
print("Measurement results:", counts)
plot_histogram(counts)
plt.show()
