using System;
using System.Collections.Generic;
using System.Linq;


namespace TSP
{
    public class SimulatedAnnealing
    {
        private Random random = new Random();

        public (List<string> path, double distance) Run(Dictionary<string, Dictionary<string, double>> graph, double initialTemperature, double coolingRate, int iterations)
        {
            var currentSolution = graph.Keys.OrderBy(c => random.Next()).ToList();
            double currentDistance = CalculateTotalDistance(graph, currentSolution);
            var bestSolution = new List<string>(currentSolution);
            double bestDistance = currentDistance;

            double temperature = initialTemperature;

            for (int i = 0; i < iterations; i++)
            {
                var newSolution = Mutate(currentSolution);
                double newDistance = CalculateTotalDistance(graph, newSolution);

                if (newDistance < bestDistance || AcceptanceProbability(currentDistance, newDistance, temperature) > random.NextDouble())
                {
                    currentSolution = new List<string>(newSolution);
                    currentDistance = newDistance;

                    if (currentDistance < bestDistance)
                    {
                        bestSolution = new List<string>(currentSolution);
                        bestDistance = currentDistance;
                    }
                }

                temperature *= coolingRate;
            }

            return (bestSolution, bestDistance);
        }

        private double AcceptanceProbability(double currentDistance, double newDistance, double temperature)
        {
            if (newDistance < currentDistance)
            {
                return 1.0;
            }
            return Math.Exp((currentDistance - newDistance) / temperature);
        }

        private List<string> Mutate(List<string> path)
        {
            int idx1 = random.Next(path.Count);
            int idx2 = random.Next(path.Count);
            var newPath = new List<string>(path);
            var temp = newPath[idx1];
            newPath[idx1] = newPath[idx2];
            newPath[idx2] = temp;
            return newPath;
        }

        private double CalculateTotalDistance(Dictionary<string, Dictionary<string, double>> graph, List<string> path)
        {
            double totalDistance = 0;
            for (int i = 0; i < path.Count - 1; i++)
            {
                totalDistance += graph[path[i]][path[i + 1]];
            }
            totalDistance += graph[path[path.Count - 1]][path[0]]; // Return to the starting city
            return totalDistance;
        }
    }
}