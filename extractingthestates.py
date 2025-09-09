import random
import cmath

def extract_qubit_state(states, amplitudes):
    """
    Select one qubit state based on quantum amplitudes.
    
    Args:
        states (list): Possible qubit states (like "|0>", "|1>", etc.)
        amplitudes (list): Complex amplitudes for each state.
    
    Returns:
        chosen_state: The extracted current state after weighted random choice.
    """
    # Step 1: Convert amplitudes -> probabilities
    probabilities = [abs(a)**2 for a in amplitudes]

    # Step 2: Normalize (in case amplitudes weren’t normalized already)
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    # Step 3: Weighted random choice
    chosen_state = random.choices(states, weights=probabilities, k=1)[0]
    return chosen_state, probabilities


# Example quantum superposition:  (1/√2)|0>  - i/√2 |1>   +  0.5|+>  + 0.5|->  
states = ["|0>", "|1>", "|+>", "|->"]
amplitudes = [1/cmath.sqrt(2), -1j/cmath.sqrt(2), 0.5, 0.5]

# Extract single current state
chosen, probs = extract_qubit_state(states, amplitudes)
print("Extracted current state:", chosen)
print("Probabilities (|amplitude|^2):")
for s, p in zip(states, probs):
    print(f"  {s}: {p:.3f}")

# Run multiple trials to show distribution matches probabilities
results = {s: 0 for s in states}
trials = 10000
for _ in range(trials):
    chosen, _ = extract_qubit_state(states, amplitudes)
    results[chosen] += 1

print("\nObserved frequencies after", trials, "trials:")
for s in states:
    print(f"{s}: {results[s]/trials:.3f}")
