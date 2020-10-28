//
// Created by Benjamin Decker on 16/10/2020.
//

#pragma once

#include "Coord3D.h"
#include "Particle.h"

typedef Kokkos::View<Coord3D *> Coord3DView;

// Change this to scale the initial distance of the particles (lower means closer together)
constexpr double scale = 1;

class ParticleContainer {
 public:
  unsigned int size;
  Coord3DView positions;
  Coord3DView forces;
  Coord3DView oldForces;
  Coord3DView velocities;

  explicit ParticleContainer(int cubeSideLength);

  [[nodiscard]] Particle getParticle(int index) const;
  void insertParticle(Particle particle, int index) const;
};
