using System;
using System.Collections.Generic;
using OfficeOpenXml;
using System.IO;
using TSP.DTO;

namespace TSP
{
    public static class GraphBuilder
    {
        public static List<CityDistance> ReadExcelFile(string filePath)
        {
            var cityDistances = new List<CityDistance>();

            ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
            using (var package = new ExcelPackage(new FileInfo(filePath)))
            {
                ExcelWorksheet worksheet = package.Workbook.Worksheets[0];
                int rowCount = worksheet.Dimension.Rows;

                for (int row = 2; row <= rowCount; row++) // Assuming first row is headers
                {
                    string city1 = worksheet.Cells[row, 1].Text;
                    string city2 = worksheet.Cells[row, 2].Text;
                    double distance = double.Parse(worksheet.Cells[row, 3].Text);

                    cityDistances.Add(new CityDistance { City1 = city1, City2 = city2, Distance = distance });
                }
            }

            return cityDistances;
        }

        public static Dictionary<string, Dictionary<string, double>> BuildGraph(List<CityDistance> cityDistances)
        {
            var graph = new Dictionary<string, Dictionary<string, double>>();

            foreach (var cityDistance in cityDistances)
            {
                if (!graph.ContainsKey(cityDistance.City1))
                {
                    graph[cityDistance.City1] = new Dictionary<string, double>();
                }

                if (!graph.ContainsKey(cityDistance.City2))
                {
                    graph[cityDistance.City2] = new Dictionary<string, double>();
                }

                graph[cityDistance.City1][cityDistance.City2] = cityDistance.Distance;
                graph[cityDistance.City2][cityDistance.City1] = cityDistance.Distance; // Assuming undirected graph
            }

            return graph;
        }
    }

}
