using System;
using System.Collections.Generic;
using System.Linq;

namespace TSP
{
    public static class NearestNeighbor
    {
        public static (List<string> path, double distance) Run(Dictionary<string, Dictionary<string, double>> graph, string startCity)
        {
            var unvisited = new HashSet<string>(graph.Keys);
            var path = new List<string>();
            var currentCity = startCity;
            double totalDistance = 0;

            path.Add(currentCity);
            unvisited.Remove(currentCity);

            while (unvisited.Count > 0)
            {
                var nearestCity = unvisited
                    .OrderBy(city => graph[currentCity][city])
                    .First();

                totalDistance += graph[currentCity][nearestCity];
                currentCity = nearestCity;
                path.Add(currentCity);
                unvisited.Remove(currentCity);
            }

            // Return to the starting city to complete the tour
            totalDistance += graph[currentCity][startCity];
            path.Add(startCity);

            return (path, totalDistance);
        }
    }
}