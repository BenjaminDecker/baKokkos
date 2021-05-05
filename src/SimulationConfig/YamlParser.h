//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>
#include <optional>
#include "../Types/Coord3D.h"
#include "ParticleCuboid.h"
#include "ParticleSphere.h"
#include "CubeClosestPacked.h"
#include "GaussianGenerator.h"

/**
 * @brief Reads a yaml file and saves its configuration data
 */
class YamlParser {
 public:
  std::optional<int> iterations;
  std::optional<float> deltaT;
  std::optional<float> cutoff;
  std::optional<std::string> vtkFileName;
  std::optional<int> vtkWriteFrequency;
  std::optional<Coord3D> globalForce;
  std::optional<std::pair<Coord3D, Coord3D>> box;
  std::vector<std::shared_ptr<const ParticleGroup>> particleGroups;
  std::optional<float> stdDev;

  explicit YamlParser(const std::string &fileName);
};
