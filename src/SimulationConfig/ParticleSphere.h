//
// Created by Benjamin Decker on 15.11.20.
//

#pragma once

#include "ParticleGroup.h"
#include "CubeClosestPacked.h"

/// Represents a sphere of particles
class ParticleSphere : public ParticleGroup {
 public:
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

//  /**
//   * This function creates a sphere by building unit sphere shells that are then scaled to be spacing distance apart,
//   * with a roughly equidistant particle distribution on them. For each sphere shell there are n(r) particles
//   * placed on it according to the the fibonacci spiral sphere algorithm.
//   * https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
//   * The algorithm takes the total amount of particles to be placed on the sphere and distributes them equally.
//   * Each particle will roughly have a distance of spacing to the next particle, if the particle density is set at
//   * 1/(sqrt(spacing)). The total amount of particles on one sphere shell is therefore: area * 1/(sqrt(spacing)).
//   */
//  [[nodiscard]] std::vector<Particle> getParticles(int startID) const override {
//    std::vector<Particle> particles;
//    int idCounter = startID;
//    particles.emplace_back(idCounter++, typeID, center, velocity);
//
//    const double gr = (sqrt(5.0) + 1.0) / 2.0;  // golden ratio = 1.6180339887498948482
//    const double ga = (2.0 - gr) * (2.0 * M_PI);  // golden angle = 2.39996322972865332
//
//    // Iterate over sphere shells of radius i * spacing
//    for (int i = 1; i * spacing <= radius; ++i) {
//      const double r = i * spacing; // Current radius
//
//      const double shellArea = r * r * M_PI;
//      const int particlesOnShell = static_cast<int>(shellArea * (1.0 / std::sqrt(spacing)));
//
//      // https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
//      for (int k = 1; k <= particlesOnShell; ++k) {
//        const double lat = asin(-1.0 + 2.0 * double(k) / (particlesOnShell + 1));
//        const double lon = ga * k;
//
//        Coord3D position = Coord3D(cos(lon) * cos(lat) * r + center.x,
//                                   sin(lon) * cos(lat) * r + center.y,
//                                   sin(lat) * r + center.z);
//
//        particles.emplace_back(idCounter++, typeID, position.rotateRollPitchYaw(i, i, i, center), velocity);
//      }
//    }
//    return particles;
//  }

  [[nodiscard]] std::vector<Particle> getParticles(int startID) const override {
    const Coord3D bottomLeftCorner = center - Coord3D(1.0, 1.0, 1.0) * (radius + 0.5 * spacing);
    const Coord3D boxLength = Coord3D(1.0, 1.0, 1.0) * (radius * 2 + spacing);
    CubeClosestPacked
        cube(typeID, spacing, velocity, particleEpsilon, particleSigma, particleMass, bottomLeftCorner, boxLength);
    auto cubeParticles = cube.getParticles();
    std::vector<Particle> sphereParticles;
    for (const auto &particle : cubeParticles) {
      if (particle.position.distanceTo(center).absoluteValue() <= radius) {
        Particle newParticle = particle;
        newParticle.particleID = startID++;
        sphereParticles.push_back(newParticle);
      }
    }
    return sphereParticles;
  }

  [[nodiscard]] std::vector<Particle> getParticles() const override {
    return getParticles(0);
  }
};
