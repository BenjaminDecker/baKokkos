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
        typeIDs(Kokkos::View<int *>("typeIDs", size)),
        positions(Coord3DView("positions", size)),
        forces(Coord3DView("forces", size)),
        oldForces(Coord3DView("oldForces", size)),
        velocities(Coord3DView("velocities", size)) {}

  [[nodiscard]] Particle getParticle(int index) const {
    int typeID = Kokkos::create_mirror_view(Kokkos::subview(typeIDs, index))();
    Coord3D position = Kokkos::create_mirror_view(Kokkos::subview(positions, index))();
    Coord3D force = Kokkos::create_mirror_view(Kokkos::subview(forces, index))();
    Coord3D oldForce = Kokkos::create_mirror_view(Kokkos::subview(oldForces, index))();
    Coord3D velocity = Kokkos::create_mirror_view(Kokkos::subview(velocities, index))();
    return Particle(typeID, position, force, velocity, oldForce);
  }

  void addParticle(const Particle &p) {
    std::cout << size << "\t" << capacity << std::endl;
    if (size == capacity) {
      capacity *= 2;
      std::cout << size << "\t" << capacity << std::endl;
      Kokkos::View<int *> newTypeIDs = Kokkos::View<int *>("typeIDs", capacity);
      Coord3DView newPositions = Coord3DView("positions", capacity);
      Coord3DView newForces = Coord3DView("forces", capacity);
      Coord3DView newOldForces = Coord3DView("oldForces", capacity);
      Coord3DView newVelocities = Coord3DView("velocities", capacity);
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(int i) {
        newTypeIDs(i) = typeIDs(i);
        newPositions(i) = positions(i);
        newForces(i) = forces(i);
        newOldForces(i) = oldForces(i);
        newVelocities(i) = velocities(i);
      });
      typeIDs = newTypeIDs;
      positions = newPositions;
      forces = newForces;
      oldForces = newOldForces;
      velocities = newVelocities;
    }
    std::cout << size << "\t" << capacity << std::endl;
    Kokkos::parallel_for("addParticle", 1, KOKKOS_LAMBDA(int i) {
      typeIDs(size) = p.typeID;
      positions(size) = p.position;
      forces(size) = p.force;
      oldForces(size) = p.oldForce;
      velocities(size) = p.velocity;
    });
    ++size;
    std::cout << size << "\t" << capacity << std::endl << std::endl;
  }
};
