//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <vector>
#include "../Types/Coord3D.h"
#include "../Types/Particle.h"

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
  const float spacing; /**< Describes particle density. Smaller values mean that particles are closer together. */
  const Coord3D velocity; /**< Starting velocity vector of all particles of the group */
  const float particleEpsilon; /**< Epsilon property of all particles of the group */
  const float particleSigma; /**< Sigma property of all particles of the group */
  const float particleMass; /**< Mass property of all particles of the group */

  ParticleGroup(int typeID,
                float spacing,
                const Coord3D velocity,
                float particleEpsilon,
                float particleSigma,
                float particleMass)
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
