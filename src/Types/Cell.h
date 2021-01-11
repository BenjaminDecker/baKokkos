//
// Created by Benjamin Decker on 19/11/2020.
//

#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include "Coord3D.h"
#include "Particle.h"

constexpr int resizeFactor = 2;

class Cell {
 public:
  int size;
  int capacity;
  const bool isHaloCell;
  const Coord3D bottomLeftCorner;

  KOKKOS_INLINE_FUNCTION
  Cell() : Cell(0, false, Coord3D()) {};

#ifdef USE_AOS
  Cell(int capacity, bool isHaloCell, Coord3D bottomLeftCorner)
      : size(0),
        isHaloCell(isHaloCell),
        bottomLeftCorner(bottomLeftCorner),
        capacity(capacity),
        particles(Kokkos::View<Particle *>(Kokkos::view_alloc("particles", Kokkos::WithoutInitializing), capacity)) {}

  void addParticle(const Particle &particle) {
    if (size == capacity) {
      capacity *= 2;
      Kokkos::resize(particles, capacity);
    }
    Kokkos::parallel_for("addParticle", 1, KOKKOS_LAMBDA(int i) {
      particles(size) = particle;
    });
    ++size;
  }

  void removeParticle(int index) {
    --size;
    Kokkos::parallel_for("removeParticle", 1, KOKKOS_LAMBDA(int i) {
      particles(index) = particles(size);
    });
  }

  [[nodiscard]] std::vector<Particle> getParticles() const {
    auto h_particles = Kokkos::create_mirror_view(particles);
    std::vector<Particle> v_particles;
    v_particles.reserve(size);
    for (int i = 0; i < size; ++i) {
      v_particles.push_back(h_particles(i));
    }
    return v_particles;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &positionAt(int index) const {
    return particles(index).position;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &velocityAt(int index) const {
    return particles(index).velocity;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &forceAt(int index) const {
    return particles(index).force;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &oldForceAt(int index) const {
    return particles(index).oldForce;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  int &particleIDAt(int index) const {
    return particles(index).particleID;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  int &typeIDAt(int index) const {
    return particles(index).typeID;
  }

 private:
  Kokkos::View<Particle *> particles;
#else

  Cell(int capacity, bool isHaloCell, Coord3D bottomLeftCorner)
      : size(0),
        isHaloCell(isHaloCell),
        bottomLeftCorner(bottomLeftCorner),
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
    Kokkos::deep_copy(Kokkos::subview(positions, size), particle.position);
    Kokkos::deep_copy(Kokkos::subview(velocities, size), particle.velocity);
    Kokkos::deep_copy(Kokkos::subview(forces, size), particle.force);
    Kokkos::deep_copy(Kokkos::subview(oldForces, size), particle.oldForce);
    Kokkos::deep_copy(Kokkos::subview(particleIDs, size), particle.particleID);
    Kokkos::deep_copy(Kokkos::subview(typeIDs, size), particle.typeID);

    ++size;
  }

  void removeParticle(int index) {
    --size;
    Kokkos::deep_copy(Kokkos::subview(positions, index), Kokkos::subview(positions, size));
    Kokkos::deep_copy(Kokkos::subview(velocities, index), Kokkos::subview(velocities, size));
    Kokkos::deep_copy(Kokkos::subview(forces, index), Kokkos::subview(forces, size));
    Kokkos::deep_copy(Kokkos::subview(oldForces, index), Kokkos::subview(oldForces, size));
    Kokkos::deep_copy(Kokkos::subview(particleIDs, index), Kokkos::subview(particleIDs, size));
    Kokkos::deep_copy(Kokkos::subview(typeIDs, index), Kokkos::subview(typeIDs, size));
  }

  [[nodiscard]] std::vector<Particle> getParticles() const {
    auto h_positions = Kokkos::create_mirror_view(positions);
    auto h_velocities = Kokkos::create_mirror_view(velocities);
    auto h_forces = Kokkos::create_mirror_view(forces);
    auto h_oldForces = Kokkos::create_mirror_view(oldForces);
    auto h_particleIDs = Kokkos::create_mirror_view(particleIDs);
    auto h_typeIDs = Kokkos::create_mirror_view(typeIDs);

    Kokkos::deep_copy(h_positions, positions);
    Kokkos::deep_copy(h_velocities, velocities);
    Kokkos::deep_copy(h_forces, forces);
    Kokkos::deep_copy(h_oldForces, oldForces);
    Kokkos::deep_copy(h_particleIDs, particleIDs);
    Kokkos::deep_copy(h_typeIDs, typeIDs);

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

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &positionAt(int index) const {
    return positions(index);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &velocityAt(int index) const {
    return velocities(index);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &forceAt(int index) const {
    return forces(index);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D &oldForceAt(int index) const {
    return oldForces(index);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  int &particleIDAt(int index) const {
    return particleIDs(index);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  int &typeIDAt(int index) const {
    return typeIDs(index);
  }

 private:
  Kokkos::View<Coord3D *> positions;
  Kokkos::View<Coord3D *> velocities;
  Kokkos::View<Coord3D *> forces;
  Kokkos::View<Coord3D *> oldForces;
  Kokkos::View<int *> particleIDs;
  Kokkos::View<int *> typeIDs;
#endif
};
