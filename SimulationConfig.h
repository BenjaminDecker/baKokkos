//
// Created by ffbde on 12/11/2020.
//

#pragma once

#include <optional>
#include "Coord3D.h"

struct SimulationConfig {
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



