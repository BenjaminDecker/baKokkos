//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <yaml-cpp/yaml.h>
#include "ParticleGroup.h"

/**
 * @brief and stores ParticleGroups from a given .yaml file
 * @see ParticleGroup
 */
class YamlParser {
 public:
  std::vector<ParticleCuboid> particleCuboids; /**< Cuboids that could be read from the .yamlFile */
  std::vector<ParticleSphere> particleSpheres; /**< Spheres that could be read from the .yamlFile */
  explicit YamlParser(const std::string &fileName); /**< Initializes the parser and starts parsing the specified file */
};



