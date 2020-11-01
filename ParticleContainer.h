//
// Created by Benjamin Decker on 16/10/2020.
//

#pragma once

#include "Coord3D.h"
#include "Particle.h"

using Coord3DView = Kokkos::View<Coord3D *>;

// Change this to scale the initial distance of the particles (lower means closer together)
constexpr double scale = 1;

/**
 * @brief Manages particle information in device space and offers reading and writing to it from the host space
 *
 * All particle information is saved in device space for performance reasons.
 * Data in device memory can be accessed from host space by copying data from device memory to host memory. This is
 * done with the Particle struct.
 *
 * @see Particle
 */
class ParticleContainer {
 public:
  unsigned int size;
  Coord3DView positions;
  Coord3DView forces;
  Coord3DView oldForces;
  Coord3DView velocities;

  /**
   * @brief Initializes a cube of particles with cubeSideLength side length.
   *
   * Particles are placed 1 unit length apart from one another. The total amount of particles initialized by the container
   * is cubeSideLength * cubeSideLength * cubeSideLength.
   */
  explicit ParticleContainer(int cubeSideLength);

  /**
   * Creates a Particle with information from the device memory at the specified index.
   */
  [[nodiscard]] Particle getParticle(int index) const;

  /**
   * Inserts the information stored in a Particle into the device memory at the specified index.
   */
  void insertParticle(Particle particle, int index) const;
};
