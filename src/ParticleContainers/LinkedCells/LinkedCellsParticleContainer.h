//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once

#include "../ParticleContainer.h"

/**
 * @brief Divides the simulation domain into cells, which contain particles.
 *
 * The cells are cubes with a sidelength of the cutoff property. This means that every particle in one cell only needs
 * to calculate the contributing force to and from other particles within the same cell and particles inside neighboring
 * cells. This automatically filters out many of the force calculations of particle pairs that are further away from
 * another than the cutoff distance and can therefore be neglected.
 * This layout improves performance if the simulation contains more than a handful of particles, as the time complexity
 * for the force calculation grows with only O(n).
 *
 * @see Particle, Coord3D
 */
class LinkedCellsParticleContainer : public ParticleContainer {
 public:

  /// Creates a Particle from the particle information in device memory with the specified id.
  [[nodiscard]] Particle getParticle(int id) const override;

  /// Inserts the information stored in a Particle into device memory with the specified id.
  void insertParticle(const Particle &particle, int id) const override;
};
