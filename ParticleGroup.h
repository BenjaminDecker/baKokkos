//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <vector>
#include "Coord3D.h"
#include "Particle.h"

/**
 * @brief Superclass for particle groups.
 *
 * Each particle group describes a layout of particles. The groups represent an efficient way to store
 * particle position, velocity and type information before initializing the simulation.
 * All particle groups share some common properties, which are stored in superclass variables.
 */
class ParticleGroup {
 public:
  const int typeID; /**< Type identifier for looking up further particle properties */
  const double spacing; /**< Describes particle density. Smaller values mean that particles are closer together. */
  const Coord3D velocity; /**< Starting velocity vector of all particles of the group */
  const double particleEpsilon; /**< Epsilon property of all particles of the group */
  const double particleSigma; /**< Sigma property of all particles of the group */
  const double particleMass; /**< Mass property of all particles of the group */

  ParticleGroup(int typeID,
                double spacing,
                const Coord3D velocity,
                double particleEpsilon,
                double particleSigma,
                double particleMass)
      : typeID(typeID),
        spacing(spacing),
        velocity(velocity),
        particleEpsilon(particleEpsilon),
        particleSigma(particleSigma),
        particleMass(particleMass) {}

  /// Creates particles based on the group parameters and returns them inside of a std::vector
  [[nodiscard]] virtual std::vector<Particle> getParticles() const = 0;

  /// Creates particles with a given start ID based on the group parameters and returns them inside of a std::vector
  [[nodiscard]] virtual std::vector<Particle> getParticles(int startID) const = 0;
};
