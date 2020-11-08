
#include <Kokkos_Core.hpp>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include "Simulation.h"
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    cxxopts::Options options("baKokkos");
    options.add_options("Non-mandatory")
        ("help", "Display this message")
        ("containerStructure",
         "Container structure that is used to store and iterate over particles. Possible values: (DirectSum LinkedCells).",
         cxxopts::value<std::string>()->default_value("DirectSum"))
        ("iterations", "Number of iterations to simulate", cxxopts::value<int>()->default_value("100000"))
        ("cutoff", "Lennard-Jones force cutoff", cxxopts::value<double>()->default_value("3"))
        ("deltaT", "Length of one time step of the simulation", cxxopts::value<double>()->default_value("0.000002"))
        ("vtk-filename", "Basename for all VTK output files", cxxopts::value<std::string>())
        ("vtk-write-frequency",
         "Number of iterations after which a VTK file is written",
         cxxopts::value<int>()->default_value("10000"))
        ("yaml-filename", "Path to a .yaml input file", cxxopts::value<std::string>());
    auto result = options.parse(argc, argv);

    if (result.count("help") > 0) {
      std::cout << options.help() << std::endl;
    } else {
      SimulationConfig::ContainerStructure containerStructure;
      int iterations;
      double deltaT;
      double cutoff;
      bool vtkOutput = false;
      std::string vtkFileName = std::string();
      int vtkWriteFrequency;
      bool yamlInput = false;
      std::string yamlFileName = std::string();

      std::string containerStructureString = result["containerStructure"].as<std::string>();
      if (containerStructureString == "DirectSum") {
        containerStructure = SimulationConfig::ContainerStructure::DirectSum;
      } else if (containerStructureString == "LinkedCells") {
        containerStructure = SimulationConfig::ContainerStructure::LinkedCells;
      } else {
        spdlog::info('"' + containerStructureString + '"'
                         + " is not a valid option for the field --containerStructure. Using "
                         + '"' + "DirectSum" + '"' + " instead.");
        containerStructure = SimulationConfig::ContainerStructure::DirectSum;
      }

      iterations = result["iterations"].as<int>();
      deltaT = result["deltaT"].as<double>();
      cutoff = result["cutoff"].as<double>();
      if (result.count("vtk-filename") > 0) {
        vtkOutput = true;
        vtkFileName = result["vtk-filename"].as<std::string>();
        vtkWriteFrequency = result["vtk-write-frequency"].as<int>();
      }
      if (result.count("yaml-filename") > 0) {
        yamlInput = true;
        yamlFileName = result["yaml-filename"].as<std::string>();
      }

      SimulationConfig config = SimulationConfig(containerStructure,
                                                 iterations,
                                                 deltaT,
                                                 cutoff,
                                                 vtkOutput,
                                                 vtkFileName,
                                                 vtkWriteFrequency,
                                                 yamlInput,
                                                 yamlFileName);

      Simulation simulation = Simulation(config);
      simulation.start();
    }
    Kokkos::finalize();
  }
  return 0;
}
