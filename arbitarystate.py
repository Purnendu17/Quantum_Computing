import numpy as np
import matplotlib.pyplot as plt

# Define basis
zero = np.array([[1], [0]], dtype=complex)
one = np.array([[0], [1]], dtype=complex)

# Arbitrary state |phi> = (1/√2)|0> - (i/√2)|1>
phi = (1/np.sqrt(2)) * zero - (1j/np.sqrt(2)) * one
phi = phi / np.linalg.norm(phi)

# Extract amplitudes
a = phi[0,0]   # coefficient of |0>
b = phi[1,0]   # coefficient of |1>

# Probabilities
prob_zero = np.abs(a)**2
prob_one = np.abs(b)**2

# Phases (in radians)
phase_zero = np.angle(a)
phase_one = np.angle(b)

print(f"|phi> = {a:.3f}|0> + {b:.3f}|1>")
print(f"Probabilities: |0>={prob_zero:.3f}, |1>={prob_one:.3f}")
print(f"Phases: |0>={phase_zero:.3f} rad, |1>={phase_one:.3f} rad")

# --- Plot probabilities ---
plt.subplot(1, 2, 1)
plt.bar(["|0>", "|1>"], [prob_zero, prob_one], color=["blue", "orange"])
plt.ylabel("Probability")
plt.title("Measurement Probabilities")

# --- Plot phases ---
plt.subplot(1, 2, 2, polar=True)
plt.polar([0, phase_zero], [0, np.abs(a)], marker='o', label="|0>")
plt.polar([0, phase_one], [0, np.abs(b)], marker='o', label="|1>")
plt.title("Amplitudes & Phases (Polar)")
plt.legend()

plt.tight_layout()
plt.show() 
