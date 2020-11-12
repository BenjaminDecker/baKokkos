//
// Created by Benjamin Decker on 10/11/2020.
//

#pragma once

#include "ParticleContainer.h"

struct LinkedCell {
  int size;
  int capacity;
  Kokkos::View<int *> typeIDs; /**< Type identifiers for looking up further particle properties */
  Coord3DView positions; /**< Array of 3-dimensional position vectors */
  Coord3DView forces; /**< Array of 3-dimensional force vectors acting on particles */
  Coord3DView oldForces; /**< Array of 3-dimensional force vectors acting on particles from the previous iteration */
  Coord3DView velocities; /**< Array of 3-dimensional velocity vectors */

  LinkedCell()
      : size(0),
        capacity(1),
        typeIDs(Kokkos::View<int *>("typeIDs", capacity)),
        positions(Coord3DView("positions", capacity)),
        forces(Coord3DView("forces", capacity)),
        oldForces(Coord3DView("oldForces", capacity)),
        velocities(Coord3DView("velocities", capacity)) {}

  [[nodiscard]] Particle getParticle(int index) const {
    int typeID = Kokkos::create_mirror_view(Kokkos::subview(typeIDs, index))();
    Coord3D position = Kokkos::create_mirror_view(Kokkos::subview(positions, index))();
    Coord3D force = Kokkos::create_mirror_view(Kokkos::subview(forces, index))();
    Coord3D oldForce = Kokkos::create_mirror_view(Kokkos::subview(oldForces, index))();
    Coord3D velocity = Kokkos::create_mirror_view(Kokkos::subview(velocities, index))();
    return Particle(typeID, position, force, velocity, oldForce);
  }

  void addParticle(const Particle &p) {
    if (size == capacity) {
      capacity *= 2;
      Kokkos::resize(typeIDs, capacity);
      Kokkos::resize(positions, capacity);
      Kokkos::resize(forces, capacity);
      Kokkos::resize(oldForces, capacity);
      Kokkos::resize(velocities, capacity);
    }
    Kokkos::parallel_for("addParticle", 1, KOKKOS_LAMBDA(int i) {
      typeIDs(size) = p.typeID;
      positions(size) = p.position;
      forces(size) = p.force;
      oldForces(size) = p.oldForce;
      velocities(size) = p.velocity;
    });
    ++size;
  }
};
