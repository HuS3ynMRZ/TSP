using System;
using System.Collections.Generic;
using System.Linq;

namespace TSP
{
    public static class GreedyAlgorithm
    {
        public static (List<string> path, double distance) Run(Dictionary<string, Dictionary<string, double>> graph)
        {
            var edges = graph.SelectMany(
                kvp => kvp.Value.Select(
                    edge => new { From = kvp.Key, To = edge.Key, Distance = edge.Value })
                ).OrderBy(e => e.Distance).ToList();

            var path = new List<string>();
            var visited = new HashSet<string>();
            double totalDistance = 0;

            foreach (var edge in edges)
            {
                if (visited.Count == graph.Count)
                {
                    path.Add(edge.From);
                    totalDistance += graph[edge.From][path[0]]; // Closing the loop
                    break;
                }

                if (!visited.Contains(edge.From))
                {
                    path.Add(edge.From);
                    visited.Add(edge.From);
                    totalDistance += edge.Distance;
                }
            }

            return (path, totalDistance);
        }
    }
}