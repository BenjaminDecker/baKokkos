//
// Created by Benjamin Decker on 16/10/2020.
//

#include "DirectSumParticleContainer.h"

DirectSumParticleContainer::DirectSumParticleContainer(const YamlParser &parser) {
  std::vector<std::vector<Particle>> cuboids;
  std::vector<std::vector<Particle>> spheres;

  for (auto &cuboid : parser.particleCuboids) {
    cuboids.emplace_back();
    cuboid.getParticles(cuboids.at(cuboids.size() - 1));
  }
  for (auto &sphere : parser.particleSpheres) {
    spheres.emplace_back();
    sphere.getParticles(spheres.at(spheres.size() - 1));
  }
  size = 0;
  for (auto &cuboid : cuboids) {
    size += cuboid.size();
  }
  for (auto &sphere : spheres) {
    size += sphere.size();
  }
  typeIDs = Kokkos::View<int *>("typeIDs", size);
  positions = Coord3DView("positions", size);
  forces = Coord3DView("forces", size);
  oldForces = Coord3DView("oldForces", size);
  velocities = Coord3DView("velocities", size);

  auto h_typeIDs = Kokkos::create_mirror_view(typeIDs);
  auto h_positions = Kokkos::create_mirror_view(positions);
  auto h_forces = Kokkos::create_mirror_view(forces);
  auto h_oldForces = Kokkos::create_mirror_view(oldForces);
  auto h_velocities = Kokkos::create_mirror_view(velocities);

  int index = 0;
  for (auto &cuboid : cuboids) {
    for (auto &particle : cuboid) {
      h_typeIDs(index) = particle.typeID;
      h_positions(index) = particle.position;
      h_forces(index) = particle.force;
      h_oldForces(index) = particle.oldForce;
      h_velocities(index) = particle.velocity;
      ++index;
    }
  }
  for (auto &sphere : spheres) {
    for (auto &particle : sphere) {
      h_typeIDs(index) = particle.typeID;
      h_positions(index) = particle.position;
      h_forces(index) = particle.force;
      h_oldForces(index) = particle.oldForce;
      h_velocities(index) = particle.velocity;
      ++index;
    }
  }

  Kokkos::deep_copy(typeIDs, h_typeIDs);
  Kokkos::deep_copy(positions, h_positions);
  Kokkos::deep_copy(forces, h_forces);
  Kokkos::deep_copy(oldForces, h_oldForces);
  Kokkos::deep_copy(velocities, h_velocities);
}

Particle DirectSumParticleContainer::getParticle(int id) const {
  int typeID = Kokkos::create_mirror_view(Kokkos::subview(typeIDs, id))();
  Coord3D position = Kokkos::create_mirror_view(Kokkos::subview(positions, id))();
  Coord3D force = Kokkos::create_mirror_view(Kokkos::subview(forces, id))();
  Coord3D oldForce = Kokkos::create_mirror_view(Kokkos::subview(oldForces, id))();
  Coord3D velocity = Kokkos::create_mirror_view(Kokkos::subview(velocities, id))();
  return Particle(typeID, position, force, velocity, oldForce);
}

void DirectSumParticleContainer::insertParticle(const Particle &particle, int id) const {
  auto subP = Kokkos::subview(positions, id);
  auto subF = Kokkos::subview(forces, id);
  auto subOF = Kokkos::subview(oldForces, id);
  auto subV = Kokkos::subview(velocities, id);

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
