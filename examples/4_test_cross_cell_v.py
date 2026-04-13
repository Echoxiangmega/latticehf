"""Debug: Test numpy assignment."""

import numpy as np

V = np.zeros((1, 1), dtype=complex)
print(f"V before: {V}")

V[0, 0] += 0.5 * 1.0
print(f"V after: {V}")

V2 = np.zeros((1, 1), dtype=complex)
V2[0, 0] = 0.5
print(f"V2: {V2}")

V3 = np.zeros((1, 1), dtype=complex)
idx = 0
V3[idx, idx] = 0.5
print(f"V3: {V3}")