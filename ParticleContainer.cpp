//
// Created by ffbde on 16/10/2020.
//

#include "ParticleContainer.h"

ParticleContainer::ParticleContainer(int cubeSideLength) {
  size = cubeSideLength * cubeSideLength * cubeSideLength;
  positions = Coord3DView("positions", size);
  forces = Coord3DView("forces", size);
  velocities = Coord3DView("velocities", size);
  Kokkos::parallel_for("initialize particles", size,
                       KOKKOS_LAMBDA(int n) {
                         positions(n) = Coord3D(n % cubeSideLength,
                                                (n / cubeSideLength) % cubeSideLength,
                                                n / (cubeSideLength * cubeSideLength));
                         forces(n) = Coord3D(0, 0, 0);
                         velocities(n) = Coord3D(0, 0, 0);
                       });
}
