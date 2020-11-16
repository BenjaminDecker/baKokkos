//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>
#include <optional>
#include "../Containers/Coord3D.h"
#include "ParticleCuboid.h"
#include "ParticleSphere.h"
#include "CubeClosestPacked.h"

/**
 * @brief and stores ParticleGroups from a given .yaml file
 * @see ParticleGroup
 */
class YamlParser {
 public:
  std::optional<int> iterations;
  std::optional<double> deltaT;
  std::optional<double> cutoff;
  std::optional<std::pair<Coord3D, Coord3D>> box;
  std::optional<std::string> vtkFileName;
  std::optional<int> vtkWriteFrequency;
  std::vector<ParticleCuboid> particleCuboids; /**< Cuboids that could be read from the .yamlFile */
  std::vector<ParticleSphere> particleSpheres; /**< Spheres that could be read from the .yamlFile */
  std::vector<CubeClosestPacked> cubesClosest; /**< Closest packed cubes that could be read from the .yamlFile */

  explicit YamlParser(const std::string &fileName);
  [[nodiscard]] std::vector<Particle> getParticles() const;
};
