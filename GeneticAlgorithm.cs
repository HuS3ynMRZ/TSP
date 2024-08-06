using System;
using System.Collections.Generic;
using System.Linq;

namespace TSP
{
    public class GeneticAlgorithm
    {
        private Random random = new Random();

        public (List<string> path, double distance) Run(Dictionary<string, Dictionary<string, double>> graph, int populationSize, int generations)
        {
            var population = InitializePopulation(graph.Keys.ToList(), populationSize);
            List<string> bestPath = null;
            double bestDistance = double.MaxValue;

            for (int i = 0; i < generations; i++)
            {
                var newPopulation = new List<List<string>>();

                foreach (var individual in population)
                {
                    var offspring = Mutate(individual);
                    newPopulation.Add(offspring);
                }

                foreach (var individual in newPopulation)
                {
                    double distance = CalculateTotalDistance(graph, individual);
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestPath = new List<string>(individual);
                    }
                }

                population = newPopulation;
            }

            return (bestPath, bestDistance);
        }

        private List<List<string>> InitializePopulation(List<string> cities, int populationSize)
        {
            var population = new List<List<string>>();
            for (int i = 0; i < populationSize; i++)
            {
                var individual = cities.OrderBy(c => random.Next()).ToList();
                population.Add(individual);
            }
            return population;
        }

        private List<string> Mutate(List<string> individual)
        {
            int idx1 = random.Next(individual.Count);
            int idx2 = random.Next(individual.Count);
            var offspring = new List<string>(individual);
            var temp = offspring[idx1];
            offspring[idx1] = offspring[idx2];
            offspring[idx2] = temp;
            return offspring;
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