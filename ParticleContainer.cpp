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

  auto subP_h = Kokkos::create_mirror_view(subP);
  auto subF_h = Kokkos::create_mirror_view(subF);
  auto subOF_h = Kokkos::create_mirror_view(subOF);
  auto subV_h = Kokkos::create_mirror_view(subV);

  subP_h() = particle.position;
  subF_h() = particle.force;
  subOF_h() = particle.oldForce;
  subV_h() = particle.velocity;

  Kokkos::deep_copy(subP, subP_h);
  Kokkos::deep_copy(subF, subF_h);
  Kokkos::deep_copy(subOF, subOF_h);
  Kokkos::deep_copy(subV, subV_h);
}
