//
// Created by Benjamin Decker on 28.10.20.
//

#pragma once

#include "Coord3D.h"

/**
 * @brief Used by a ParticleContainer to represent, view and change particle information in device memory from host space
 *
 * 3-dimensional vectors are saved via the Coord3D struct.
 *
 * @see ParticleContainer, Coord3D
 */
struct Particle {
  int typeID; /**< Type identifier for looking up further particle properties */
  Coord3D position; /**< 3-dimensional position vector */
  Coord3D force; /**< 3-dimensional force vector acting on particle */
  Coord3D oldForce; /**< 3-dimensional force vector acting on particle from the previous iteration */
  Coord3D velocity; /**< 3-dimensional velocity vector */


  explicit Particle(int typeID) : typeID(typeID) {}
  Particle(int typeID, Coord3D position) : typeID(typeID), position(position) {}
  Particle(int typeID, Coord3D position, Coord3D velocity) : typeID(typeID), position(position), velocity(velocity) {}
  Particle(int typeID, Coord3D position, Coord3D force, Coord3D velocity, Coord3D oldForce)
      : typeID(typeID), position(position), force(force), velocity(velocity), oldForce(oldForce) {}
};



