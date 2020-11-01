//
// Created by Benjamin Decker on 28.10.20.
//

#pragma once

#include "Coord3D.h"

/**
 * @brief Used by a ParticleContainer to view and change particle information in device memory from the host space
 */
struct Particle {
  Coord3D position;
  Coord3D force;
  Coord3D oldForce;
  Coord3D velocity;
};



