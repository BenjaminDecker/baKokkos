//
// Created by Benjamin Decker on 12/11/2020.
//

#pragma once

#include <optional>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include "../Types/Coord3D.h"
#include "YamlParser.h"

/**
 * The SimulationConfig class is given to the constructor of the Simulation class. It contains all information needed
 * to construct a Simulation. It can either be constructed by specifying information in its constructor or by reading
 * the command line and parsing a yaml file.
 */
class SimulationConfig {
 public:
  /// Structure type that is used to store and iterate over particles
  const enum ContainerStructure { DirectSum, LinkedCells } containerStructure;
  /// Number of iterations to simulate
  const int iterations;
  /// Length of one time step of the simulation
  const float deltaT;
  /// Maximum distance between two particles for which the force calculation can not be neglected to increase performance
  const float cutoff;
  /// Global force acting on every particle in every time step
  const Coord3D globalForce;
  /// Pair of front-left-bottom corner and back-right-top corner of the simulation space cuboid
  const std::optional<const std::pair<const Coord3D, const Coord3D>> box;
  /// Optional pair of a vtk file basename and a vtk write frequency for the vtk file output
  const std::optional<const std::pair<const std::string, const int>> vtk;
  /**
   * The particles for the simulation are given in a compact form as particle groups. These groups offer calculating
   * and returning the particles contained in the group.
   * @see ParticleGroup
   */
  const std::vector<std::shared_ptr<const ParticleGroup>> particleGroups;

  SimulationConfig(const ContainerStructure container_structure,
                   const int iterations,
                   const float delta_t,
                   const float cutoff,
                   const Coord3D &global_force,
                   const std::optional<const std::pair<const Coord3D, const Coord3D>> &box,
                   const std::optional<const std::pair<const std::string, const int>> &vtk,
                   const std::vector<std::shared_ptr<const ParticleGroup>> &particleGroups)
      : containerStructure(container_structure),
        iterations(iterations),
        deltaT(delta_t),
        cutoff(cutoff),
        globalForce(global_force),
        box(box),
        vtk(vtk),
        particleGroups(particleGroups) {}

  static SimulationConfig readConfig(int argc, char *argv[]) {
    constexpr auto helpStr = "help";
    constexpr auto containerStructureStr = "containerStructure";
    constexpr auto iterationsStr = "iterations";
    constexpr auto deltaTStr = "deltaT";
    constexpr auto cutoffStr = "cutoff";
    constexpr auto vtk_filenameStr = "vtk-filename";
    constexpr auto vtk_write_frequencyStr = "vtk-write-frequency";
    constexpr auto yaml_filenameStr = "yaml-filename";

    cxxopts::Options options("baKokkos");
    options.add_options(
        "If an option is not given, a default value is used. If a yaml file is used as input, every option will be overwritten by the yaml file.")
        (helpStr, "Display this message")
        (containerStructureStr,
         "Structure type that is used to store and iterate over particles. Possible values: (DirectSum LinkedCells).",
         cxxopts::value<std::string>()->default_value("LinkedCells"))
        (iterationsStr, "Number of iterations to simulate", cxxopts::value<int>()->default_value("100000"))
        (deltaTStr, "Length of one time step of the simulation", cxxopts::value<float>()->default_value("0.000002"))
        (cutoffStr,
         "Maximum distance between two particles for which the force calculation can not be neglected to increase performance",
         cxxopts::value<float>()->default_value("3"))
        (vtk_filenameStr, "Basename for all VTK output files", cxxopts::value<std::string>())
        (vtk_write_frequencyStr,
         "Number of iterations after which a VTK file is written",
         cxxopts::value<int>()->default_value("10000"))
        (yaml_filenameStr, "Path to a .yaml input file", cxxopts::value<std::string>());
    auto result = options.parse(argc, argv);

    if (result.count(helpStr) > 0) {
      std::cout << options.help() << std::endl;
      exit(0);
    }
    ContainerStructure containerStructure;
    int iterations;
    float deltaT;
    float cutoff;
    Coord3D globalForce;
    std::optional<const std::pair<const Coord3D, const Coord3D>> box;
    std::optional<const std::pair<const std::string, const int>> vtk;
    std::vector<std::shared_ptr<const ParticleGroup>> particleGroups;

    std::string containerStructureParam = result[containerStructureStr].as<std::string>();
    if (containerStructureParam == "DirectSum") {
      containerStructure = DirectSum;
    } else if (containerStructureParam == "LinkedCells") {
      containerStructure = LinkedCells;
    } else {
      spdlog::warn('"' + containerStructureParam + '"'
                       + " is not a valid option for --containerStructure. Using "
                       + '"' + "LinkedCells" + '"' + " instead.");
      containerStructure = LinkedCells;
    }
    iterations = result[iterationsStr].as<int>();
    deltaT = result[deltaTStr].as<float>();
    cutoff = result[cutoffStr].as<float>();
    if (result[vtk_filenameStr].count() > 0) {
      vtk.emplace(result[vtk_filenameStr].as<std::string>(), result[vtk_write_frequencyStr].as<int>());
    }
    if (result[yaml_filenameStr].count() > 0) {
      YamlParser parser(result[yaml_filenameStr].as<std::string>());
      if (parser.iterations) {
        iterations = parser.iterations.value();
      }
      if (parser.deltaT) {
        deltaT = parser.deltaT.value();
      }
      if (parser.cutoff) {
        cutoff = parser.cutoff.value();
      }
      if (parser.vtkFileName && parser.vtkWriteFrequency) {
        vtk.emplace(parser.vtkFileName.value(), parser.vtkWriteFrequency.value());
      }
      if (parser.globalForce) {
        globalForce = parser.globalForce.value();
      }
      if (parser.box) {
        box.emplace(parser.box.value());
      }
      particleGroups = parser.particleGroups;
    }
    return SimulationConfig(containerStructure, iterations, deltaT, cutoff, globalForce, box, vtk, particleGroups);
  }
};

static std::ostream &operator<<(std::ostream &stream, const SimulationConfig &obj) {
  stream << "ContainerStructure: ";
  switch (obj.containerStructure) {
    case SimulationConfig::DirectSum:stream << "DirectSum";
      break;
    case SimulationConfig::LinkedCells:stream << "LinkedCells";
      break;
  }
  stream << "\n";
  stream << "iterations: " << obj.iterations << "\n";
  stream << "deltaT: " << obj.deltaT << "\n";
  stream << "cutoff: " << obj.cutoff << "\n";
  stream << "globalForce: " << obj.globalForce << "\n";
  if (obj.box) {
    stream << "box:" << std::endl;
    stream << "  box-min: " << obj.box.value().first << "\n";
    stream << "  box-max: " << obj.box.value().second << "\n";
  }
  if (obj.vtk) {
    stream << "vtk-filename: " << obj.vtk.value().first << ".vtk" << "\n";
    stream << "vtk-write-frequency: " << obj.vtk.value().second << "\n";
  }
  stream.flush();
  return stream;
}