
#include <Kokkos_Core.hpp>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <optional>
#include "YamlParser.h"
#include "Simulation.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    cxxopts::Options options("baKokkos");
    options.add_options(
        "If an option is not given, a default value is used. If a yaml file is used as input, every option will be overwritten.")
        ("help", "Display this message")
        ("containerStructure",
         "Container structure that is used to store and iterate over particles. Possible values: (DirectSum LinkedCells).",
         cxxopts::value<std::string>()->default_value("DirectSum"))
        ("iterations", "Number of iterations to simulate", cxxopts::value<int>()->default_value("100000"))
        ("deltaT", "Length of one time step of the simulation", cxxopts::value<double>()->default_value("0.000002"))
        ("cutoff", "Lennard-Jones force cutoff", cxxopts::value<double>()->default_value("3"))
        ("vtk-filename", "Basename for all VTK output files", cxxopts::value<std::string>())
        ("vtk-write-frequency",
         "Number of iterations after which a VTK file is written",
         cxxopts::value<int>()->default_value("10000"))
        ("yaml-filename", "Path to a .yaml input file", cxxopts::value<std::string>());
    auto result = options.parse(argc, argv);

    if (result.count("help") > 0) {
      std::cout << options.help() << std::endl;
    } else {
      // Properties from the command line
      SimulationConfig::ContainerStructure containerStructure; /**< Represents which container structure is used to store and iterate over particles */
      int iterations; /**< Number of iterations to simulate */
      double deltaT; /**< Length of one time step of the simulation */
      double cutoff;
      std::optional<std::pair<Coord3D, Coord3D>> box;
      std::optional<std::string> vtkFileName; /**< Basename for all VTK output files */
      int vtkWriteFrequency; /**< Number of iterations after which a VTK file is written */
      std::optional<std::string> yamlFileName; /**< Path to the.yaml file used as input */
      std::vector<Particle> particles;

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
        vtkFileName = result["vtk-filename"].as<std::string>();
      }
      vtkWriteFrequency = result["vtk-write-frequency"].as<int>();

      // Read yaml options if there are any
      if (result.count("yaml-filename") > 0) {
        yamlFileName = result["yaml-filename"].as<std::string>();
      }
      if (yamlFileName) {
        YamlParser parser(yamlFileName.value());
        auto yamlIterations = parser.iterations;
        if(yamlIterations) {
          iterations = yamlIterations.value();
        }
        auto yamlDeltaT = parser.deltaT;
        if(yamlDeltaT) {
          deltaT = yamlDeltaT.value();
        }
        auto yamlCutoff = parser.cutoff;
        if(yamlCutoff) {
          cutoff = yamlCutoff.value();
        }
        auto yamlBox = parser.box;
        if(yamlBox) {
          box = yamlBox.value();
        }
        auto yamlVtkFileName = parser.vtkFileName;
        if(yamlVtkFileName) {
          vtkFileName = yamlVtkFileName.value();
        }
        auto yamlVtkWriteFrequency = parser.vtkWriteFrequency;
        if(yamlVtkWriteFrequency) {
          vtkWriteFrequency = yamlVtkWriteFrequency.value();
        }
        for (auto &cuboid : parser.particleCuboids) {
          cuboid.getParticles(particles);
        }
        for (auto &sphere : parser.particleSpheres) {
          sphere.getParticles(particles);
        }
      }

      // TODO add particles from command line. For now it is only possible to add particles from .yaml files

      const SimulationConfig config(containerStructure, iterations, deltaT, cutoff, box, vtkFileName, vtkWriteFrequency, yamlFileName);

      Simulation simulation = Simulation(config, particles);
      simulation.start();
    }
  }
  Kokkos::finalize();
  return 0;
}
