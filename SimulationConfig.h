//
// Created by ffbde on 12/11/2020.
//

#pragma once

#include <optional>
#include "Coord3D.h"

class SimulationConfig {
 public:
  const enum ContainerStructure { DirectSum, LinkedCells }
      containerStructure; /**< Represents which container structure is used to store and iterate over particles */
  const int iterations; /**< Number of iterations to simulate */
  const double deltaT; /**< Length of one time step of the simulation */
  const double cutoff;
  const std::optional<std::pair<Coord3D, Coord3D>> box;
  const std::optional<std::string> vtkFileName; /**< Basename for all VTK output files */
  const int vtkWriteFrequency; /**< Number of iterations after which a VTK file is written */
  const std::optional<std::string> yamlFileName; /**< Path to the.yaml file used as input */

  SimulationConfig(const ContainerStructure container_structure,
                   const int iterations,
                   const double delta_t,
                   const double cutoff,
                   const std::optional<std::pair<Coord3D, Coord3D>> box,
                   const std::optional<std::string> vtk_file_name,
                   const int vtk_write_frequency,
                   const std::optional<std::string> yaml_file_name
  )
      : containerStructure(container_structure),
        iterations(iterations),
        deltaT(delta_t),
        cutoff(cutoff),
        box(box),
        vtkFileName(vtk_file_name),
        vtkWriteFrequency(vtk_write_frequency),
        yamlFileName(yaml_file_name) {}
};

static std::ostream &operator<<(std::ostream &stream, const SimulationConfig &obj) {
  stream << "ContainerStructure: ";
  switch (obj.containerStructure) {
    case SimulationConfig::DirectSum:
      stream << "DirectSum";
      break;
    case SimulationConfig::LinkedCells:
      stream << "LinkedCells";
      break;
  }
  stream << std::endl;
  stream << "iterations: " << obj.iterations << std::endl;
  stream << "deltaT: " << obj.deltaT << std::endl;
  stream << "cutoff: " << obj.cutoff << std::endl;
  if(obj.box) {
    stream << "box:" << std::endl;
    stream << "  box-min: " << obj.box.value().first << std::endl;
    stream << "  box-max: " << obj.box.value().second << std::endl;
  }
  if(obj.vtkFileName) {
    stream << "vtk-filename: " << obj.vtkFileName.value() << ".vtk" << std::endl;
    stream << "vtk-write-frequency: " << obj.vtkWriteFrequency << std::endl;
  }
  return stream;
}
