//
// Created by Benjamin Decker on 15.11.20.
//

#pragma once

#include "ParticleGroup.h"

/**
 * @brief Represents a cuboid structure made up of evenly placed particles in a grid layout.
 */
class ParticleCuboid : public ParticleGroup {
 public:

  /**
   * Position of the bottom left corner of the cuboid
   */
  const Coord3D bottomLeftCorner;

  /**
   * Number of particles per dimension. Together with the spacing properties, it desctibes the cuboid sidelengths
   */
  const Coord3D particlesPerDimension;

  ParticleCuboid(int typeID,
                 double spacing,
                 Coord3D velocity,
                 double particleEpsilon,
                 double particleSigma,
                 double particleMass,
                 Coord3D bottomLeftCorner,
                 Coord3D particlesPerDimension)
      : ParticleGroup(typeID, spacing, velocity, particleEpsilon, particleSigma, particleMass),
        bottomLeftCorner(bottomLeftCorner),
        particlesPerDimension(particlesPerDimension) {}

  /// Places particles in a cuboid grid with dimensions based on the particlesPerDimension and the spacing property.
  // TODO rename to generateParticles
  [[nodiscard]] std::vector<Particle> getParticles(int startID = 0) const override {
    std::vector<Particle> particles;
    int idCounter = startID;
    for (int x = 0; x < particlesPerDimension.x; ++x) {
      for (int y = 0; y < particlesPerDimension.y; ++y) {
        for (int z = 0; z < particlesPerDimension.z; ++z) {
          Coord3D position = Coord3D(bottomLeftCorner.x + x * spacing,
                                     bottomLeftCorner.y + y * spacing,
                                     bottomLeftCorner.z + z * spacing);
          particles.emplace_back(idCounter++, typeID, position, velocity);
        }
      }
    }
    return particles;
  }

  // TODO default parameter
  [[nodiscard]] std::vector<Particle> getParticles() const override {
    return getParticles(0);
  }
};
