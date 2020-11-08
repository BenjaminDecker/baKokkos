//
// Created by Benjamin Decker on 16/10/2020.
//

#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include "../ParticleContainer.h"
#include "../../Helper/Coord3D.h"
#include "../../Helper/Particle.h"
#include "../../Yaml/YamlParser.h"

using Coord3DView = Kokkos::View<Coord3D *>;

/**
 * @brief Uses no special layout. Computations are simple, but performance is bad.
 *
 * Trivial layout approach. Every particle is saved inside the same Kokkos::View.
 * Iterating over the particles is simple, but there is no easy way to filter out particle pairs that are further apart
 * from another than the cutoff distance.
 * This layout should only be used if the simulation contains only a handful of particles, as the time complexity for
 * the force calculation grows with O(n^2).
 *
 * @see Particle, Coord3D
 */
class DirectSumParticleContainer : public ParticleContainer {
 public:
  unsigned int size; /**< Number of saved particles */
  Kokkos::View<int *> typeIDs; /**< Type identifiers for looking up further particle properties */
  Coord3DView positions; /**< Array of 3-dimensional position vectors */
  Coord3DView forces; /**< Array of 3-dimensional force vectors acting on particles */
  Coord3DView oldForces; /**< Array of 3-dimensional force vectors acting on particles from the previous iteration */
  Coord3DView velocities; /**< Array of 3-dimensional velocity vectors */

  /**
   * @brief Initialises particles
   * @param parser stores particle information from a .yaml file
   */
  explicit DirectSumParticleContainer(const YamlParser &parser);
  DirectSumParticleContainer() = default;

  /// Creates a Particle from the particle information in device memory with the specified id.
  [[nodiscard]] Particle getParticle(int id) const override;

  /// Inserts the information stored in a Particle into device memory with the specified id.
  void insertParticle(const Particle &particle, int id) const override;
};
