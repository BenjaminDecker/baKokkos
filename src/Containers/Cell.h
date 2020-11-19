//
// Created by Benjamin Decker on 19/11/2020.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "Coord3D.h"
#include "Particle.h"

constexpr int resizeFactor = 2;
constexpr int numNeighbours = 26;

class Cell {
 public:
  int size;
  int capacity;
  Kokkos::View<Coord3D *> positions;
  Kokkos::View<Coord3D *> velocities;
  Kokkos::View<Coord3D *> forces;
  Kokkos::View<Coord3D *> oldForces;
  Kokkos::View<int *> particleIDs;
  Kokkos::View<int *> typeIDs;

  KOKKOS_INLINE_FUNCTION
  Cell() : Cell(0) {};

  KOKKOS_INLINE_FUNCTION
  explicit Cell(int capacity)
      : size(0),
        capacity(capacity),
        positions(Kokkos::View<Coord3D *>(Kokkos::view_alloc("positions", Kokkos::WithoutInitializing), capacity)),
        velocities(Kokkos::View<Coord3D *>(Kokkos::view_alloc("velocities", Kokkos::WithoutInitializing), capacity)),
        forces(Kokkos::View<Coord3D *>(Kokkos::view_alloc("forces", Kokkos::WithoutInitializing), capacity)),
        oldForces(Kokkos::View<Coord3D *>(Kokkos::view_alloc("oldForces", Kokkos::WithoutInitializing), capacity)),
        particleIDs(Kokkos::View<int *>(Kokkos::view_alloc("particleIDs", Kokkos::WithoutInitializing), capacity)),
        typeIDs(Kokkos::View<int *>(Kokkos::view_alloc("typeIDs", Kokkos::WithoutInitializing), capacity)) {}

  /**
   * Adds a particle to the cell by appending its property information at the ends of the separate property views.
   * If necessary, this also resizes the cell to double its capacity.
   * This function should only be called from host space.
   */
  void addParticle(const Particle &particle) {
    if (size == capacity) {
      capacity *= 2;
      Kokkos::resize(positions, capacity);
      Kokkos::resize(velocities, capacity);
      Kokkos::resize(forces, capacity);
      Kokkos::resize(oldForces, capacity);
      Kokkos::resize(particleIDs, capacity);
      Kokkos::resize(typeIDs, capacity);
    }
    Kokkos::parallel_for("addParticle", 1, KOKKOS_LAMBDA(int i) {
      positions(size) = particle.position;
      velocities(size) = particle.velocity;
      forces(size) = particle.force;
      oldForces(size) = particle.oldForce;
      particleIDs(size) = particle.particleID;
      typeIDs(size) = particle.typeID;
    });
    ++size;
  }

  void removeParticle(int index) {
    --size;
    Kokkos::parallel_for("removeParticle", 1, KOKKOS_LAMBDA(int i) {
      positions(index) = positions(size);
      velocities(index) = velocities(size);
      forces(index) = forces(size);
      oldForces(index) = oldForces(size);
      particleIDs(index) = particleIDs(size);
      typeIDs(index) = typeIDs(size);
    });
  }

  [[nodiscard]] std::vector<Particle> getParticles() const {
    auto h_positions = Kokkos::create_mirror_view(positions);
    auto h_velocities = Kokkos::create_mirror_view(velocities);
    auto h_forces = Kokkos::create_mirror_view(forces);
    auto h_oldForces = Kokkos::create_mirror_view(oldForces);
    auto h_particleIDs = Kokkos::create_mirror_view(particleIDs);
    auto h_typeIDs = Kokkos::create_mirror_view(typeIDs);
    std::vector<Particle> particles;
    particles.reserve(size);
    for (int i = 0; i < size; ++i) {
      particles.emplace_back(h_particleIDs(i),
                             h_typeIDs(i),
                             h_positions(i),
                             h_forces(i),
                             h_velocities(i),
                             h_oldForces(i));
    }
    return particles;
  }
};
