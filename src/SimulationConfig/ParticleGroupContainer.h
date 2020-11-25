//
// Created by Benjamin Decker on 25.11.20.
//

#pragma once

#include <vector>
#include "ParticleSphere.h"
#include "ParticleCuboid.h"
#include "CubeClosestPacked.h"

class ParticleGroupContainer {
  const std::vector<const std::shared_ptr<const ParticleGroup>> particleGroups;
};
