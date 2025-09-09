from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# Create a quantum circuit with 1 qubit
qc = QuantumCircuit(1)

# Step 1: Apply X gate to flip |0> to |1>
qc.x(0)

# Step 2: Apply H gate to get (|0> - |1>)/sqrt(2)
qc.h(0)

# Display the circuit
print(qc.draw())

# Simulate the statevector to verify
sim = Aer.get_backend('statevector_simulator')
result = execute(qc, sim).result()
statevector = result.get_statevector()

print("Final statevector:", statevector)
