//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once
#include "Particle.h"

using Coord3DView = Kokkos::View<Coord3D *>;

/**
 * @brief Manages particle information in device memory and offers reading and writing to it from the host space.
 *
 * All particle information is saved in device space for performance reasons. All information is stored in a
 * structure-of-arrays layout, split up into one Kokkos::View for every particle property.
 * Data in device memory can be accessed from host space by copying data from device memory to host memory. This is
 * done via the Particle struct.
 * 3-dimensional vectors are saved via the Coord3D struct.
 */
class ParticleContainer {
 public:
  unsigned int size; /**< Number of saved particles */

  /// Creates a Particle from the particle information in device memory with the specified id.
  [[nodiscard]] virtual Particle getParticle(int id) const = 0;

  /// Inserts the information stored in a Particle into device memory with the specified id.
  virtual void insertParticle(const Particle &particle, int id) const = 0;
};



