using System;
using System.Collections.Generic;
using System.IO.Pipes;

namespace TSP
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = "path_to_your_excel_file.xlsx";
            var cityDistances = GraphBuilder.ReadExcelFile(filePath);
            var graph = GraphBuilder.BuildGraph(cityDistances);

            string startCity = "CityName"; // Specify your starting city

            // Nearest Neighbor
            var (nnPath, nnDistance) = NearestNeighbor.Run(graph, startCity);
            Console.WriteLine("Nearest Neighbor Path: " + string.Join(" -> ", nnPath));
            Console.WriteLine("Distance: " + nnDistance);

            // Greedy Algorithm
            var (gaPath, gaDistance) = GreedyAlgorithm.Run(graph);
            Console.WriteLine("Greedy Algorithm Path: " + string.Join(" -> ", gaPath));
            Console.WriteLine("Distance: " + gaDistance);

            // Genetic Algorithm
            var geneticAlgorithm = new GeneticAlgorithm();
            var (gaBestPath, gaBestDistance) = geneticAlgorithm.Run(graph, populationSize: 50, generations: 100);
            Console.WriteLine("Genetic Algorithm Path: " + string.Join(" -> ", gaBestPath));
            Console.WriteLine("Distance: " + gaBestDistance);

            // Simulated Annealing
            var simulatedAnnealing = new SimulatedAnnealing();
            var (saPath, saDistance) = simulatedAnnealing.Run(graph, initialTemperature: 10000, coolingRate: 0.995, iterations: 1000);
            Console.WriteLine("Simulated Annealing Path: " + string.Join(" -> ", saPath));
            Console.WriteLine("Distance: " + saDistance);

            // Ant Colony Optimization
            var antColonyOptimization = new AntColonyOptimization();
            var (acoPath, acoDistance) = antColonyOptimization.Run(graph, antCount: 20, iterations: 100, alpha: 1.0, beta: 5.0, evaporationRate: 0.5);
            Console.WriteLine("Ant Colony Optimization Path: " + string.Join(" -> ", acoPath));
            Console.WriteLine("Distance: " + acoDistance);
        }
    }

}
