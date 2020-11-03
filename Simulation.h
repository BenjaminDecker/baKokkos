//
// Created by ffbde on 03/11/2020.
//

#pragma once

#include <cxxopts.hpp>

class Simulation {
 public:
  int iterations; /**< Amount of timesteps to simulate */
  double deltaT; /**< Length of one time step of the simulation */

  bool vtkOutput = false; /**< Indicates if the user specified a vtk file name as output. */
  std::string vtkFileName; /**< Base name of all vtk output files. */

  bool yamlInput = false; /**< Indicates if the user specified a yaml file path as input. */
  std::string yamlFileName; /**< Path to the.yaml file used as input */

  const double epsilon = 1;
  const double sigma = 1;
  const double mass = 1;

  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  Simulation(int argc, char *argv[]);
};



