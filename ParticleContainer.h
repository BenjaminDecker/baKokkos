//
// Created by Benjamin Decker on 16/10/2020.
//

#pragma once

#include "Coord3D.h"

typedef Kokkos::View<Coord3D *> Coord3DView;

class ParticleContainer {
 public:
  int size;
  Coord3DView positions;
  Coord3DView forces;
  Coord3DView velocities;

  explicit ParticleContainer(int cubeSideLength) {
    size = cubeSideLength * cubeSideLength * cubeSideLength;
    positions = Coord3DView("positions", size);
    forces = Coord3DView("forces", size);
    velocities = Coord3DView("velocities", size);

    Kokkos::parallel_for("initializeParticles", size, KOKKOS_LAMBDA(int n) {
      positions(n) = Coord3D(n % cubeSideLength,
                             (n / cubeSideLength) % cubeSideLength,
                             n / (cubeSideLength * cubeSideLength));
      forces(n) = velocities(n) = Coord3D();
    });
  }
};
