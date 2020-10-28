//
// Created by Benjamin Decker on 28.10.20.
//

#pragma once

#include "Coord3D.h"

struct Particle {
  Coord3D position;
  Coord3D force;
  Coord3D oldForce;
  Coord3D velocity;
};



