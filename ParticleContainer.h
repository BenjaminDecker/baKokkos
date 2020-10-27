//
// Created by Benjamin Decker on 16/10/2020.
//

#pragma once

#include "Coord3D.h"

typedef Kokkos::View<Coord3D *> Coord3DView;

// Change this to scale the initial distance of the particles
constexpr double scale = 1;

class ParticleContainer {
 public:
  unsigned int size;
  Coord3DView positions;
  Coord3DView forces;
  Coord3DView oldForces;
  Coord3DView velocities;

  explicit ParticleContainer(int cubeSideLength) {
    size = cubeSideLength * cubeSideLength * cubeSideLength + 2;
    positions = Coord3DView("positions", size);
    forces = Coord3DView("forces", size);
    oldForces = Coord3DView ("oldForces", size);
    velocities = Coord3DView("velocities", size);

    Kokkos::parallel_for("initializeParticles", size, KOKKOS_LAMBDA(int n) {
      positions(n) = Coord3D(n % cubeSideLength,
                             (n / cubeSideLength) % cubeSideLength,
                             n / (cubeSideLength * cubeSideLength));

      forces(n) = oldForces(n) = velocities(n) = Coord3D();
    });

    positions(size - 2) = Coord3D(0.5, 0.5, 2);
    positions(size - 1) = Coord3D(0.5, 0.5, -1);

    Kokkos::parallel_for("scalePositions", size, KOKKOS_LAMBDA(int n) {
      positions(n) *= scale;
    });

  }
};
