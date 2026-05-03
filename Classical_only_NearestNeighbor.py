import numpy as np
import matplotlib.pyplot as plt

# city generator
np.random.seed(42) # <------------------------- Lock the randomness here
n_cities = 10
coords = np.random.randint(0, 100, size=(n_cities, 2))

# distance matrix
distance_matrix = np.array([
    [np.linalg.norm(coords[i] - coords[j]) for j in range(n_cities)]
    for i in range(n_cities)
])

print("Coordinates:")
for i, (x, y) in enumerate(coords):
    print(f"  City {i}: ({x}, {y})")

print("\nDistance Matrix:")
print(np.round(distance_matrix, 2))

# nearest neighbor solve
def total_distance(route, dist_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i+1]]
    total += dist_matrix[route[-1]][route[0]]  # return to start
    return total

def nearest_neighbor(dist_matrix, start=0):
    n = len(dist_matrix)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)

    while unvisited:
        current = route[-1]
        # pick the closest unvisited city
        nearest = min(unvisited, key=lambda city: dist_matrix[current][city])
        route.append(nearest)
        unvisited.remove(nearest)

    return route

best_route = nearest_neighbor(distance_matrix, start=0)
best_dist = total_distance(best_route, distance_matrix)

full_route = best_route + [best_route[0]]
print(f"\nBest route: {full_route}")
print(f"Total distance: {best_dist:.2f}")

# plot for all connections
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

# plot for winner
ax2 = axes[1]
ax2.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (x, y) in enumerate(coords):
    ax2.annotate(f'City {i}', (x, y), textcoords="offset points", xytext=(10, 5), fontsize=10)

for i in range(len(full_route) - 1):
    a, b = full_route[i], full_route[i+1]
    ax2.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], 'g-', linewidth=2.5)
    mid_x = (coords[a][0] + coords[b][0]) / 2
    mid_y = (coords[a][1] + coords[b][1]) / 2
    dist = distance_matrix[a][b]
    ax2.annotate(f'{dist:.1f}', (mid_x, mid_y), fontsize=7, color='darkgreen')

# add arrows for direction
for i in range(len(full_route) - 1):
    a, b = full_route[i], full_route[i+1]
    ax2.annotate('', xy=coords[b], xytext=coords[a],
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax2.set_title(f'Best Route — Distance: {best_dist:.2f}')
ax2.grid(True)

plt.suptitle('TSP — 4 Cities (Nearest Neighbor)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()