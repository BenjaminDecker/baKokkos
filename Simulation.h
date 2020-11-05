//
// Created by ffbde on 03/11/2020.
//

#pragma once

#include <iomanip>
#include <fstream>
#include <utility>
#include "ParticleContainer.h"

/**
 * @brief This struct stores configuration information for the simulation. It is needed to create a Simulation object.
 *
 * @see Simulation
 */
struct SimulationConfig {
  const int iterations; /**< Number of iterations to simulate */
  const double deltaT; /**< Length of one time step of the simulation */

  const bool vtkOutput = false; /**< Indicates if the user specified a vtk file name as output */
  const std::string vtkFileName; /**< Basename for all VTK output files */

  const int vtkWriteFrequency; /**< Number of iterations after which a VTK file is written */

  const bool yamlInput = false; /**< Indicates if the user specified a yaml file path as input */
  const std::string yamlFileName; /**< Path to the.yaml file used as input */

  SimulationConfig(int iterations,
                   double delta_t,
                   bool vtk_output,
                   std::string vtk_file_name,
                   int vtk_write_frequency,
                   bool yaml_input,
                   std::string yaml_file_name)
      : iterations(iterations),
        deltaT(delta_t),
        vtkOutput(vtk_output),
        vtkFileName(std::move(vtk_file_name)),
        vtkWriteFrequency(vtk_write_frequency),
        yamlInput(yaml_input),
        yamlFileName(std::move(yaml_file_name)) {}
};


/**
 * @brief This class controls the simulation
 */
class Simulation {
 public:
  ParticleContainer container; /**< Stores and manages particle data in device memory */

  const SimulationConfig config; /**< Stores configuration information for the simulation */

  //TODO get those from the ParticlePropertiesLibrary
  const double epsilon = 1;
  const double sigma = 1;
  const double mass = 1;

  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  /// The simulation is initialized by parsing the command line input for parameters
  Simulation(const SimulationConfig& config);

  /// Starts the simulation
  void start() const;

  /**
   * Writes a .vtk file about the current state of the simulation
   * @param Current iteration
   */
  void writeVTKFile(int iteration) const;
};



