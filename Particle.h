//
// Created by Benjamin Decker on 28.10.20.
//

#pragma once

#include "Coord3D.h"

/**
 * @brief Used by a ParticleContainer to view and change particle information in device memory from the host space
 *
 * @see ParticleContainer
 */
struct Particle {
  int typeID;
  Coord3D position;
  Coord3D force;
  Coord3D velocity;
  Coord3D oldForce;

  explicit Particle(int typeID) : typeID(typeID) {}
  Particle(int typeID, Coord3D position) : typeID(typeID), position(position) {}
  Particle(int typeID, Coord3D position, Coord3D velocity) : typeID(typeID), position(position), velocity(velocity) {}
  Particle(int typeID, Coord3D position, Coord3D force, Coord3D velocity, Coord3D oldForce)
      : typeID(typeID), position(position), force(force), velocity(velocity), oldForce(oldForce) {}
};



