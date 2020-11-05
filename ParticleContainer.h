//
// Created by Benjamin Decker on 16/10/2020.
//

#pragma once

#include "YamlParser.h"

using Coord3DView = Kokkos::View<Coord3D *>;

/**
 * @brief Manages particle information in device space and offers reading and writing to it from the host space.
 *
 * All particle information is saved in device space for performance reasons. All information is stored in a
 * structure-of-arrays layout, split up into one Kokkos::View for every particle property.\n
 * Data in device memory can be accessed from host space by copying data from device memory to host memory. This is
 * done via the Particle struct.\n
 * 3-dimensional vectors are saved via the Coord3D struct.
 *
 * @see Particle, Coord3D
 */
class ParticleContainer {
 public:
  unsigned int size; /**< Number of saved particles */
  Kokkos::View<int *> typeIDs; /**< Type identifiers for looking up further particle properties */
  Coord3DView positions; /**< Array of 3-dimensional position vectors */
  Coord3DView forces; /**< Array of 3-dimensional force vectors acting on particles */
  Coord3DView oldForces; /**< Array of 3-dimensional force vectors acting on particles from the previous iteration */
  Coord3DView velocities; /**< Array of 3-dimensional velocity vectors */

  /**
   * @brief Initializes a cube of particles with cubeSideLength side length.
   *
   * Particles are placed 1 unit length apart from one another. The total amount of particles initialized by the
   * container is cubeSideLength * cubeSideLength * cubeSideLength.
   */
  explicit ParticleContainer(int cubeSideLength);

  /**
   * @brief Initialises particles
   * @param parser stores particle information from a .yaml file
   */
  explicit ParticleContainer(const YamlParser &parser);
  ParticleContainer() = default;

  /**
   * Creates a Particle with information from device memory at the specified index.
   */
  [[nodiscard]] Particle getParticle(int index) const;

  /**
   * Inserts the information stored in a Particle into device memory at the specified index.
   */
  void insertParticle(const Particle &particle, int index) const;
};
