import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time
import psutil
import os

# ── HARDWARE MONITOR SETUP ────────────────────────────────────────────
process = psutil.Process(os.getpid())

def get_hardware_stats():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_used = process.memory_info().rss / 1024 / 1024  # MB
    ram_total = psutil.virtual_memory().total / 1024 / 1024  # MB
    return cpu_percent, ram_used, ram_total

# ── CITY GENERATOR ────────────────────────────────────────────────────
np.random.seed(57)
n_cities = 12
coords = np.random.randint(0, 100, size=(n_cities, 2))

# ── DISTANCE MATRIX ───────────────────────────────────────────────────
distance_matrix = np.array([
    [np.linalg.norm(coords[i] - coords[j]) for j in range(n_cities)]
    for i in range(n_cities)
])

print("Coordinates:")
for i, (x, y) in enumerate(coords):
    print(f"  City {i}: ({x}, {y})")

print("\nDistance Matrix:")
print(np.round(distance_matrix, 2))

# ── BRUTE FORCE SOLVE ─────────────────────────────────────────────────
def total_distance(route, dist_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i+1]]
    total += dist_matrix[route[-1]][route[0]]
    return total

print("\n" + "="*60)
print("CLASSICAL BENCHMARK")
print("="*60)

cpu_before_c, ram_before_c, ram_total = get_hardware_stats()
time_classical_start = time.perf_counter()

cities = list(range(1, n_cities))
best_route = None
best_dist = float('inf')

for perm in permutations(cities):
    route = [0] + list(perm)
    dist = total_distance(route, distance_matrix)
    if dist < best_dist:
        best_dist = dist
        best_route = route

time_classical_end = time.perf_counter()
cpu_after_c, ram_after_c, _ = get_hardware_stats()

classical_time = time_classical_end - time_classical_start

full_route = best_route + [0]
print(f"Best route:            {full_route}")
print(f"Total distance:        {best_dist:.2f}")
print(f"Time taken:            {classical_time:.6f} seconds")
print(f"CPU usage:             {cpu_after_c:.1f}%")
print(f"RAM used:              {ram_after_c:.1f} MB / {ram_total:.1f} MB")
print(f"Routes checked:        {len(list(permutations(range(1, n_cities))))}")

# ── CLASSICAL PLOTS ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (x, y) in enumerate(coords):
    ax1.annotate(f'City {i}', (x, y), textcoords="offset points", xytext=(10, 5), fontsize=10)
for i in range(n_cities):
    for j in range(i+1, n_cities):
        x_vals = [coords[i][0], coords[j][0]]
        y_vals = [coords[i][1], coords[j][1]]
        ax1.plot(x_vals, y_vals, 'b-', alpha=0.3)
        mid_x = (coords[i][0] + coords[j][0]) / 2
        mid_y = (coords[i][1] + coords[j][1]) / 2
        dist = np.linalg.norm(coords[i] - coords[j])
        ax1.annotate(f'{dist:.1f}', (mid_x, mid_y), fontsize=7, color='gray')
ax1.set_title('All Connections')
ax1.grid(True)

ax2 = axes[1]
ax2.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (x, y) in enumerate(coords):
    ax2.annotate(f'City {i}', (x, y), textcoords="offset points", xytext=(10, 5), fontsize=10)
for i in range(len(full_route) - 1):
    a, b = full_route[i], full_route[i+1]
    ax2.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], 'g-', linewidth=2.5)
    mid_x = (coords[a][0] + coords[b][0]) / 2
    mid_y = (coords[a][1] + coords[b][1]) / 2
    ax2.annotate(f'{distance_matrix[a][b]:.1f}', (mid_x, mid_y), fontsize=7, color='darkgreen')
for i in range(len(full_route) - 1):
    a, b = full_route[i], full_route[i+1]
    ax2.annotate('', xy=coords[b], xytext=coords[a],
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.set_title(f'Classical Best Route — Distance: {best_dist:.2f}')
ax2.grid(True)

plt.suptitle('TSP — Classical Brute Force', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ====================================================================
# ── QUANTUM TSP WITH GROVER'S ALGORITHM ─────────────────────────────
# ====================================================================

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# ── 1. ENUMERATE ALL ROUTES ───────────────────────────────────────────
all_routes = [[0] + list(p) for p in permutations(range(1, n_cities))]

def route_distance(route):
    return sum(distance_matrix[route[t]][route[(t+1) % len(route)]] for t in range(len(route)))

route_distances = [(i, route, route_distance(route)) for i, route in enumerate(all_routes)]
route_distances.sort(key=lambda x: x[2])

n_qubits_needed = int(np.ceil(np.log2(len(all_routes) + 1)))
print("\nAll possible routes:")
for i, route, dist in route_distances:
    print(f"  |{format(i, f'0{n_qubits_needed}b')}⟩ Route {i}: {route + [route[0]]} — distance: {dist:.2f}")

target_idx = route_distances[0][0]
target_route = route_distances[0][1]
target_dist = route_distances[0][2]
target_state = format(target_idx, f'0{n_qubits_needed}b')

print(f"\nTarget state: |{target_state}⟩ — Grover's should find this")

# ── 2. BUILD GROVER'S CIRCUIT ─────────────────────────────────────────
n_routes = len(all_routes)
n_qubits = int(np.ceil(np.log2(n_routes + 1)))

# Calculate optimal number of Grover iterations
n_iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(n_routes))))
print(f"Qubits needed:         {n_qubits}")
print(f"Grover iterations:     {n_iterations}")

qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Step 1: Hadamard on all qubits
circuit.h(qr)
circuit.barrier()

# Step 2 & 3: Repeat oracle + diffusion for optimal iterations
for iteration in range(n_iterations):

    # Oracle — phase flip the target state
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            circuit.x(qr[i])
    circuit.h(qr[n_qubits - 1])
    circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    circuit.h(qr[n_qubits - 1])
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            circuit.x(qr[i])
    circuit.barrier()

    # Diffusion operator
    circuit.h(qr)
    circuit.x(qr)
    circuit.h(qr[n_qubits - 1])
    circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    circuit.h(qr[n_qubits - 1])
    circuit.x(qr)
    circuit.h(qr)
    circuit.barrier()

# Step 4: Measure
circuit.measure(qr, cr)
# ── 3. RUN ON SIMULATOR ───────────────────────────────────────────────
print("\n" + "="*60)
print("QUANTUM BENCHMARK")
print("="*60)

cpu_before_q, ram_before_q, _ = get_hardware_stats()
time_quantum_start = time.perf_counter()

simulator = AerSimulator()
job = simulator.run(circuit, shots=1000)
result = job.result()
counts = result.get_counts()

time_quantum_end = time.perf_counter()
cpu_after_q, ram_after_q, _ = get_hardware_stats()

quantum_time = time_quantum_end - time_quantum_start

print(f"Time taken:            {quantum_time:.6f} seconds")
print(f"CPU usage:             {cpu_after_q:.1f}%")
print(f"RAM used:              {ram_after_q:.1f} MB / {ram_total:.1f} MB")
print(f"Qubits used:           {n_qubits}")
print(f"Shots:                 1000")


# ── 4. DECODE RESULT ──────────────────────────────────────────────────
valid_counts = {k: v for k, v in counts.items() if int(k, 2) < len(all_routes)}
measured = max(valid_counts, key=valid_counts.get)
measured_idx = int(measured, 2)
quantum_route = all_routes[measured_idx]
q_dist = route_distance(quantum_route)

print(f"\nGrover's found route:  {quantum_route + [quantum_route[0]]}")
print(f"Quantum distance:      {q_dist:.2f}")
print(f"Classical distance:    {best_dist:.2f}")

if abs(q_dist - best_dist) < 0.01:
    print("✓ Grover's matched the optimal solution!")
else:
    print("✗ Grover's found a suboptimal solution")

# ── 5. TIMING SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Classical time:      {classical_time:.6f} seconds")
print(f"  Quantum time:        {quantum_time:.6f} seconds")
print(f"  Speedup:             {classical_time/quantum_time:.2f}x  ({'quantum faster' if quantum_time < classical_time else 'classical faster'})")
print(f"  Classical RAM:       {ram_before_c:.1f} → {ram_after_c:.1f} MB")
print(f"  Quantum RAM:         {ram_before_q:.1f} → {ram_after_q:.1f} MB")
print(f"  Classical routes:    {len(all_routes)} checked one by one")
print(f"  Quantum routes:      {len(all_routes)} searched simultaneously")
print("="*60)

# ── 6. PLOT RESULTS ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax1 = axes[0]
ax1.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (cx, cy) in enumerate(coords):
    ax1.annotate(f'City {i}', (cx, cy), textcoords="offset points", xytext=(10, 5), fontsize=10)
for i in range(n_cities):
    for j in range(i+1, n_cities):
        ax1.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 'b-', alpha=0.3)
        mid_x = (coords[i][0] + coords[j][0]) / 2
        mid_y = (coords[i][1] + coords[j][1]) / 2
        ax1.annotate(f'{distance_matrix[i][j]:.1f}', (mid_x, mid_y), fontsize=7, color='gray')
ax1.set_title('All Connections')
ax1.grid(True)

ax2 = axes[1]
ax2.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (cx, cy) in enumerate(coords):
    ax2.annotate(f'City {i}', (cx, cy), textcoords="offset points", xytext=(10, 5), fontsize=10)
full_classical = best_route + [best_route[0]]
for i in range(len(full_classical)-1):
    a, b = full_classical[i], full_classical[i+1]
    ax2.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], 'g-', linewidth=2.5)
    ax2.annotate('', xy=coords[b], xytext=coords[a],
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.set_title(f'Classical Brute Force\nDistance: {best_dist:.2f}\nTime: {classical_time:.6f}s')
ax2.grid(True)

ax3 = axes[2]
ax3.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (cx, cy) in enumerate(coords):
    ax3.annotate(f'City {i}', (cx, cy), textcoords="offset points", xytext=(10, 5), fontsize=10)
full_quantum = quantum_route + [quantum_route[0]]
for i in range(len(full_quantum)-1):
    a, b = full_quantum[i], full_quantum[i+1]
    ax3.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], 'b-', linewidth=2.5)
    ax3.annotate('', xy=coords[b], xytext=coords[a],
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax3.set_title(f"Grover's Algorithm\nDistance: {q_dist:.2f}\nTime: {quantum_time:.6f}s")
ax3.grid(True)

plt.suptitle("TSP — Classical vs Grover's", fontsize=14, fontweight='bold')
plt.tight_layout()

# Measurement histogram
fig2, ax4 = plt.subplots(figsize=(8, 4))
valid_states = {k: v for k, v in counts.items() if int(k, 2) < len(all_routes)}
ax4.bar(
    [f"|{k}⟩" for k in valid_states.keys()],
    valid_states.values(),
    color=['green' if k == target_state else 'gray' for k in valid_states.keys()]
)
ax4.set_title("Grover's Measurement Results\n(green = correct answer)")
ax4.set_ylabel('Count (out of 1000 shots)')
ax4.grid(True, axis='y')
plt.tight_layout()

plt.show()