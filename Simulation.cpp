//
// Created by ffbde on 03/11/2020.
//

#include "Simulation.h"
Simulation::Simulation(int argc, char **argv) {
  cxxopts::Options options("baKokkos");
  options.add_options("Non-mandatory")
      ("help", "Display this message.")
      ("iterations", "Number of iterations to simulate.", cxxopts::value<int>()->default_value("100000"))
      ("deltaT", "Length of one time step of the simulation.",cxxopts::value<double>()->default_value("0.000002"))
      ("vtk-filename", "Basename for all VTK output files.", cxxopts::value<std::string>())
      ("vtk-write-frequency", "Number of iterations after which a VTK file is written.", cxxopts::value<int>()->default_value("10000"))
      ("yaml-filename", "Path to a .yaml input file.", cxxopts::value<std::string>());
  auto result = options.parse(argc, argv);
  iterations = result["iterations"].as<int>();
  deltaT = result["deltaT"].as<double>();
  if (result.count("vtk-filename") > 0) {
    vtkOutput = true;
    vtkFileName = result["vtk-filename"].as<std::string>();
  }
  if (result.count("yaml-filename") > 0) {
    yamlInput = true;
    yamlFileName = result["yaml-filename"].as<std::string>();
  }
}
