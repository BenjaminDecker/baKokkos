//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include "Particle.h"

struct ParticleGroup {
  const int typeID;
  const double spacing;
  const Coord3D velocity;
  const double particleEpsilon;
  const double particleSigma;
  const double particleMass;
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

        /**
         * The particleGroup creates particles based on its parameters and appends them to the given vector.
         */
  virtual void getParticles(std::vector<Particle> &particles) const  = 0;
};

struct ParticleCuboid : public ParticleGroup {
  const Coord3D bottomLeftCorner;
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

  void getParticles(std::vector<Particle> &particles) const override {
    for (int x = 0; x < particlesPerDimension.x; ++x) {
      for (int y = 0; y < particlesPerDimension.y; ++y) {
        for (int z = 0; z < particlesPerDimension.z; ++z) {
          Coord3D position = Coord3D(bottomLeftCorner.x + x * spacing,
                                     bottomLeftCorner.y + y * spacing,
                                     bottomLeftCorner.z + z * spacing);
          particles.emplace_back(typeID, position, velocity);
        }
      }
    }
  }
};

struct ParticleSphere : public ParticleGroup {
  const Coord3D center;
  const double radius;
  ParticleSphere(int typeID,
                 double spacing,
                 Coord3D velocity,
                 double particleEpsilon,
                 double particleSigma,
                 double particleMass,
                 Coord3D center,
                 double radius)
      : ParticleGroup(typeID, spacing, velocity, particleEpsilon, particleSigma, particleMass),
        center(center),
        radius(radius) {}

  /**
   * This function creates a sphere by building unit sphere shells that are then scaled to be spacing distance apart,
   * with a roughly equidistant particle distribution on them. For each sphere shell there are n(r) particles
   * placed on it according to the the fibonacci spiral sphere algorithm.
   * https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
   * The algorithm takes the total amount of particles to be placed on the sphere and distributes them equally.
   * Each particle will roughly have a distance of spacing to the next particle, if the particle density is set at
   * 1/(sqrt(spacing)). The total amount of particles on one sphere shell is therefore: area * 1/(sqrt(spacing)).
   */
  void getParticles(std::vector<Particle> &particles) const override {
    particles.emplace_back(typeID, center, velocity);

    const double gr = (sqrt(5.0) + 1.0) / 2.0;  // golden ratio = 1.6180339887498948482
    const double ga = (2.0 - gr) * (2.0 * M_PI);  // golden angle = 2.39996322972865332

    // Iterate over sphere shells of radius i * spacing
    for (int i = 0; i * spacing <= radius; ++i) {

      const double r = i * spacing; // Current radius

      const double shellArea = r * r * M_PI;
      const int particlesOnShell = static_cast<int>(shellArea * (1 / std::sqrt(spacing)));

      // https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
      for (int k = 1; k <= particlesOnShell; ++k) {
        const double lat = asin(-1.0 + 2.0 * double(k) / (particlesOnShell + 1));
        const double lon = ga * k;

        const double x = cos(lon) * cos(lat) * r + center.x;
        const double y = sin(lon) * cos(lat) * r + center.y;
        const double z = sin(lat) * r + center.z;

        Coord3D position = Coord3D(cos(lon) * cos(lat) * r + center.x,
                                   sin(lon) * cos(lat) * r + center.y,
                                   sin(lat) * r + center.z);

        particles.emplace_back(typeID, position, velocity);
      }

    }
  }
};