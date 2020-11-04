//
// Created by ffbde on 03/11/2020.
//

#pragma once

#include <iomanip>
#include <fstream>
#include <cxxopts.hpp>
#include "ParticleContainer.h"

class Simulation {
 public:
  int iterations; /**< Number of iterations to simulate */
  double deltaT; /**< Length of one time step of the simulation */

  bool vtkOutput = false; /**< Indicates if the user specified a vtk file name as output */
  std::string vtkFileName; /**< Basename for all VTK output files */

  int vtkWriteFrequency;

  bool yamlInput = false; /**< Indicates if the user specified a yaml file path as input */
  std::string yamlFileName; /**< Path to the.yaml file used as input */

  ParticleContainer container; /**< Holds and manages particle data in device memory  */

  const double epsilon = 1;
  const double sigma = 1;
  const double mass = 1;

  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  /**
   * The simulation is initialized by parsing the command line input for parameters.
   */
  Simulation(int argc, char *argv[]);

  /**
   * Starts the simulation.
   */
  void start() const;

  void writeVTKFile(int iteration) const;
};



