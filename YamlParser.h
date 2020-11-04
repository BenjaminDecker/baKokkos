//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <yaml-cpp/yaml.h>
#include "ParticleGroup.h"

class YamlParser {
 public:
  std::vector<ParticleCuboid> particleCuboids;
  std::vector<ParticleSphere> particleSpheres;
  explicit YamlParser(const std::string &fileName);
};



