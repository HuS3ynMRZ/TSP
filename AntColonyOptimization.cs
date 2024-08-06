using System;
using System.Collections.Generic;
using System.Linq;


namespace TSP
{
    public class AntColonyOptimization
    {
        private Random random = new Random();
        private Dictionary<string, Dictionary<string, double>> pheromones;

        public (List<string> path, double distance) Run(Dictionary<string, Dictionary<string, double>> graph, int antCount, int iterations, double alpha, double beta, double evaporationRate)
        {
            InitializePheromones(graph);

            List<string> bestPath = null;
            double bestDistance = double.MaxValue;

            for (int i = 0; i < iterations; i++)
            {
                for (int ant = 0; ant < antCount; ant++)
                {
                    var (path, distance) = ConstructSolution(graph, alpha, beta);

                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestPath = path;
                    }

                    UpdatePheromones(path, distance);
                }

                EvaporatePheromones(evaporationRate);
            }

            return (bestPath, bestDistance);
        }

        private void InitializePheromones(Dictionary<string, Dictionary<string, double>> graph)
        {
            pheromones = graph.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.ToDictionary(edge => edge.Key, edge => 1.0));
        }

        private (List<string> path, double distance) ConstructSolution(Dictionary<string, Dictionary<string, double>> graph, double alpha, double beta)
        {
            var path = new List<string>();
            var unvisited = new HashSet<string>(graph.Keys);
            string currentCity = unvisited.OrderBy(c => random.Next()).First();
            path.Add(currentCity);
            unvisited.Remove(currentCity);
            double totalDistance = 0;

            while (unvisited.Count > 0)
            {
                string nextCity = SelectNextCity(graph, currentCity, unvisited, alpha, beta);
                totalDistance += graph[currentCity][nextCity];
                currentCity = nextCity;
                path.Add(currentCity);
                unvisited.Remove(currentCity);
            }

            totalDistance += graph[currentCity][path[0]]; // Return to the starting city
            path.Add(path[0]);

            return (path, totalDistance);
        }

        private string SelectNextCity(Dictionary<string, Dictionary<string, double>> graph, string currentCity, HashSet<string> unvisited, double alpha, double beta)
        {
            var probabilities = new Dictionary<string, double>();

            double sum = 0;
            foreach (var city in unvisited)
            {
                double pheromoneLevel = pheromones[currentCity][city];
                double visibility = 1.0 / graph[currentCity][city];
                double probability = Math.Pow(pheromoneLevel, alpha) * Math.Pow(visibility, beta);
                probabilities[city] = probability;
                sum += probability;
            }

            double randomPoint = random.NextDouble() * sum;
            double cumulativeProbability = 0.0;

            foreach (var city in probabilities.Keys)
            {
                cumulativeProbability += probabilities[city];
                if (randomPoint <= cumulativeProbability)
                {
                    return city;
                }
            }

            return unvisited.First(); // Fallback (should never happen)
        }

        private void UpdatePheromones(List<string> path, double distance)
        {
            for (int i = 0; i < path.Count - 1; i++)
            {
                string fromCity = path[i];
                string toCity = path[i + 1];
                pheromones[fromCity][toCity] += 1.0 / distance;
                pheromones[toCity][fromCity] += 1.0 / distance;
            }
        }

        private void EvaporatePheromones(double evaporationRate)
        {
            foreach (var city in pheromones.Keys)
            {
                foreach (var neighbor in pheromones[city].Keys)
                {
                    pheromones[city][neighbor] *= (1 - evaporationRate);
                }
            }
        }
    }
}