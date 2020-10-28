//
// Created by Benjamin Decker on 16/10/2020.
//

#include "ParticleContainer.h"
ParticleContainer::ParticleContainer(int cubeSideLength) {
  size = cubeSideLength * cubeSideLength * cubeSideLength + 2;
  positions = Coord3DView("positions", size);
  forces = Coord3DView("forces", size);
  oldForces = Coord3DView("oldForces", size);
  velocities = Coord3DView("velocities", size);

  Kokkos::parallel_for("initializeParticles", size, KOKKOS_LAMBDA(int n) {
    positions(n) = Coord3D(n % cubeSideLength,
                           (n / cubeSideLength) % cubeSideLength,
                           n / (cubeSideLength * cubeSideLength));
  });

  positions(size - 2) = Coord3D(0.5, 0.5, 2);
  positions(size - 1) = Coord3D(0.5, 0.5, -1);

  Kokkos::parallel_for("scalePositions", size, KOKKOS_LAMBDA(int n) {
    positions(n) *= scale;
  });
}

Particle ParticleContainer::getParticle(int index) const {
  Coord3D position = Kokkos::create_mirror_view(Kokkos::subview(positions, index))();
  Coord3D force = Kokkos::create_mirror_view(Kokkos::subview(forces, index))();
  Coord3D oldForce = Kokkos::create_mirror_view(Kokkos::subview(oldForces, index))();
  Coord3D velocity = Kokkos::create_mirror_view(Kokkos::subview(velocities, index))();
  return Particle{position, force, oldForce, velocity};
}

void ParticleContainer::insertParticle(Particle particle, int index) const {
  auto subP = Kokkos::subview(positions, index);
  auto subF = Kokkos::subview(forces, index);
  auto subOF = Kokkos::subview(oldForces, index);
  auto subV = Kokkos::subview(velocities, index);

  auto h_subP = Kokkos::create_mirror_view(subP);
  auto h_subF = Kokkos::create_mirror_view(subF);
  auto h_subOF = Kokkos::create_mirror_view(subOF);
  auto h_subV = Kokkos::create_mirror_view(subV);

  h_subP() = particle.position;
  h_subF() = particle.force;
  h_subOF() = particle.oldForce;
  h_subV() = particle.velocity;

  Kokkos::deep_copy(subP, h_subP);
  Kokkos::deep_copy(subF, h_subF);
  Kokkos::deep_copy(subOF, h_subOF);
  Kokkos::deep_copy(subV, h_subV);
}
