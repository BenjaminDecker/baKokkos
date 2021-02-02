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
class Particle {
 public:
  int particleID{}; /**< Unique particle identifier */
  int typeID{}; /**< Type identifier for looking up further particle properties */
  Coord3D position; /**< 3-dimensional position vector */
  Coord3D force; /**< 3-dimensional force vector acting on particle */
  Coord3D oldForce; /**< 3-dimensional force vector acting on particle from the previous iteration */
  Coord3D velocity; /**< 3-dimensional velocity vector */

  Particle() = default;

  KOKKOS_INLINE_FUNCTION
  explicit Particle(int particleID, int typeID) : particleID(particleID), typeID(typeID) {}

  KOKKOS_INLINE_FUNCTION
  Particle(int particleID, int typeID, Coord3D position) : particleID(particleID), typeID(typeID), position(position) {}

  KOKKOS_INLINE_FUNCTION
  Particle(int particleID, int typeID, Coord3D position, Coord3D velocity)
      : particleID(particleID), typeID(typeID), position(position), velocity(velocity) {}

  KOKKOS_INLINE_FUNCTION
  Particle(int particleID, int typeID, Coord3D position, Coord3D force, Coord3D velocity, Coord3D oldForce)
      : particleID(particleID),
        typeID(typeID),
        position(position),
        force(force),
        velocity(velocity),
        oldForce(oldForce) {}

};
