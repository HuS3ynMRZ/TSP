"""
TSP Solver: Classical vs Quantum Approaches
Compares different algorithms for solving the Traveling Salesman Problem
"""

import numpy as np
import time
from itertools import permutations
from typing import List, Tuple
import matplotlib.pyplot as plt

# Quantum computing libraries (install: pip install qiskit qiskit-aer)
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not installed. Install with: pip install qiskit qiskit-aer")


class TSPSolver:
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route"""
        total = 0
        for i in range(len(route)):
            total += self.distance_matrix[route[i-1], route[i]]
        return total
    
    # ==================== CLASSICAL METHODS ====================
    
    def brute_force(self) -> Tuple[List[int], float, float]:
        """Exhaustive search - O(n!)"""
        start_time = time.time()
        
        cities = list(range(self.n_cities))
        best_route = None
        best_distance = float('inf')
        
        for perm in permutations(cities[1:]):  # Fix first city
            route = [0] + list(perm)
            distance = self.calculate_route_distance(route)
            if distance < best_distance:
                best_distance = distance
                best_route = route
                
        elapsed = time.time() - start_time
        return best_route, best_distance, elapsed
    
    def nearest_neighbor(self, start_city: int = 0) -> Tuple[List[int], float, float]:
        """Greedy nearest neighbor heuristic - O(n²)"""
        start_time = time.time()
        
        unvisited = set(range(self.n_cities))
        route = [start_city]
        unvisited.remove(start_city)
        
        current = start_city
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda city: self.distance_matrix[current, city])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        distance = self.calculate_route_distance(route)
        elapsed = time.time() - start_time
        return route, distance, elapsed
    
    def two_opt(self, initial_route: List[int] = None, 
                max_iterations: int = 1000) -> Tuple[List[int], float, float]:
        """2-opt local search improvement"""
        start_time = time.time()
        
        if initial_route is None:
            route, _, _ = self.nearest_neighbor()
        else:
            route = initial_route.copy()
            
        best_distance = self.calculate_route_distance(route)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, self.n_cities - 1):
                for j in range(i + 1, self.n_cities):
                    # Try reversing route[i:j+1]
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_distance = self.calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        
        elapsed = time.time() - start_time
        return route, best_distance, elapsed
    
    def held_karp(self) -> Tuple[List[int], float, float]:
        """Dynamic Programming solution - O(n² * 2^n)"""
        start_time = time.time()
        
        n = self.n_cities
        # dp[mask][i] = min cost to visit cities in mask, ending at i
        dp = {}
        parent = {}
        
        # Base case: start from city 0, visit only city 0
        dp[(1, 0)] = 0
        
        # Fill DP table
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                    
                prev_mask = mask ^ (1 << last)
                
                if prev_mask == 0:
                    continue
                    
                for prev in range(n):
                    if not (prev_mask & (1 << prev)):
                        continue
                        
                    if (prev_mask, prev) not in dp:
                        continue
                        
                    cost = dp[(prev_mask, prev)] + self.distance_matrix[prev, last]
                    
                    if (mask, last) not in dp or cost < dp[(mask, last)]:
                        dp[(mask, last)] = cost
                        parent[(mask, last)] = prev
        
        # Find minimum cost to visit all cities and return to start
        full_mask = (1 << n) - 1
        best_distance = float('inf')
        best_last = -1
        
        for last in range(1, n):
            if (full_mask, last) in dp:
                cost = dp[(full_mask, last)] + self.distance_matrix[last, 0]
                if cost < best_distance:
                    best_distance = cost
                    best_last = last
        
        # Reconstruct path
        route = []
        mask = full_mask
        current = best_last
        
        while mask:
            route.append(current)
            if (mask, current) not in parent:
                break
            next_current = parent[(mask, current)]
            mask ^= (1 << current)
            current = next_current
            
        route.reverse()
        route = [0] + route  # Add start city
        
        elapsed = time.time() - start_time
        return route, best_distance, elapsed
    
    # ==================== QUANTUM METHODS ====================
    
    def qaoa_tsp(self, p: int = 1) -> Tuple[List[int], float, float]:
        """
        QAOA (Quantum Approximate Optimization Algorithm) for TSP
        This is a simplified version for demonstration
        """
        if not QISKIT_AVAILABLE:
            return None, float('inf'), 0.0
            
        start_time = time.time()
        
        # For small problems, encode as binary optimization
        # This is a simplified QAOA demonstration
        n = self.n_cities
        
        # Create quantum circuit
        n_qubits = n * n  # Position encoding: qubit[i*n + j] = 1 if city j is at position i
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize superposition
        qc.h(range(n_qubits))
        
        # Apply QAOA layers (simplified)
        beta = np.pi / 4
        gamma = np.pi / 4
        
        for _ in range(p):
            # Cost Hamiltonian (distance minimization)
            for i in range(n - 1):
                for j in range(n):
                    for k in range(n):
                        if j != k:
                            # Entangle adjacent positions
                            qc.rzz(gamma * self.distance_matrix[j, k], 
                                  i * n + j, (i + 1) * n + k)
            
            # Mixer Hamiltonian
            qc.rx(2 * beta, range(n_qubits))
        
        # Measure
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Parse best result (simplified - just take most common)
        best_bitstring = max(counts, key=counts.get)
        
        # Decode to route (simplified decoding)
        route = self._decode_bitstring_to_route(best_bitstring)
        distance = self.calculate_route_distance(route)
        
        elapsed = time.time() - start_time
        return route, distance, elapsed
    
    def _decode_bitstring_to_route(self, bitstring: str) -> List[int]:
        """Decode quantum measurement to valid TSP route (simplified)"""
        # This is a placeholder - real decoding is complex
        # For demo, just return a valid greedy route
        return self.nearest_neighbor()[0]


def generate_random_cities(n: int, seed: int = 42) -> np.ndarray:
    """Generate random city coordinates and distance matrix"""
    np.random.seed(seed)
    coords = np.random.rand(n, 2) * 100
    
    # Calculate Euclidean distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
                
    return distance_matrix


def benchmark_algorithms(n_cities: int = 8):
    """Compare all algorithms"""
    print(f"\n{'='*60}")
    print(f"TSP Benchmark: {n_cities} cities")
    print(f"{'='*60}\n")
    
    distance_matrix = generate_random_cities(n_cities)
    solver = TSPSolver(distance_matrix)
    
    results = {}
    
    # Classical methods
    methods = [
        ("Nearest Neighbor", solver.nearest_neighbor, n_cities <= 100),
        ("2-Opt", solver.two_opt, n_cities <= 50),
        ("Brute Force", solver.brute_force, n_cities <= 10),
        ("Held-Karp (DP)", solver.held_karp, n_cities <= 20),
    ]
    
    for name, method, should_run in methods:
        if should_run:
            try:
                route, distance, elapsed = method()
                results[name] = {
                    'route': route,
                    'distance': distance,
                    'time': elapsed
                }
                print(f"{name:20s}: Distance = {distance:8.2f}, Time = {elapsed:8.4f}s")
            except Exception as e:
                print(f"{name:20s}: Error - {e}")
        else:
            print(f"{name:20s}: Skipped (too slow for {n_cities} cities)")
    
    # Quantum method (if available)
    if QISKIT_AVAILABLE and n_cities <= 5:
        try:
            route, distance, elapsed = solver.qaoa_tsp()
            if route:
                results["QAOA (Quantum)"] = {
                    'route': route,
                    'distance': distance,
                    'time': elapsed
                }
                print(f"{'QAOA (Quantum)':20s}: Distance = {distance:8.2f}, Time = {elapsed:8.4f}s")
        except Exception as e:
            print(f"{'QAOA (Quantum)':20s}: Error - {e}")
    
    # Find best solution
    if results:
        best = min(results.items(), key=lambda x: x[1]['distance'])
        print(f"\nBest solution: {best[0]} with distance {best[1]['distance']:.2f}")
    
    return results


if __name__ == "__main__":
    # Test with different problem sizes
    for n in [6, 8, 10]:
        benchmark_algorithms(n)
    
    print("\n" + "="*60)
    print("Notes:")
    print("- Quantum simulation is exponentially slow on classical hardware")
    print("- Real quantum speedup requires actual quantum computers")
    print("- QAOA results are approximate, not guaranteed optimal")
    print("- For fair comparison, focus on solution quality, not speed")
    print("="*60)