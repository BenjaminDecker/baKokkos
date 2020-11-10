//
// Created by Benjamin Decker on 10/11/2020.
//

#pragma once

#include <Kokkos_UnorderedMap.hpp>
#include "ParticleContainer.h"

struct LinkedCell {
  int size;
  int capacity;
  Kokkos::UnorderedMap<int, int>
      changes; /**< Saves if particles left this cell during the last iteration and their new cell number */
  Kokkos::View<int *> typeIDs; /**< Type identifiers for looking up further particle properties */
  Coord3DView positions; /**< Array of 3-dimensional position vectors */
  Coord3DView forces; /**< Array of 3-dimensional force vectors acting on particles */
  Coord3DView oldForces; /**< Array of 3-dimensional force vectors acting on particles from the previous iteration */
  Coord3DView velocities; /**< Array of 3-dimensional velocity vectors */

  LinkedCell()
      : size(0),
        capacity(0),
        changes(Kokkos::UnorderedMap<int, int>(capacity)),
        typeIDs(Kokkos::View<int *>("typeIDs", size)),
        positions(Coord3DView("positions", size)),
        forces(Coord3DView("forces", size)),
        oldForces(Coord3DView("oldForces", size)),
        velocities(Coord3DView("velocities", size)) {}

  explicit LinkedCell(int capacity)
      : size(0),
        capacity(capacity),
        changes(Kokkos::UnorderedMap<int, int>(capacity)),
        typeIDs(Kokkos::View<int *>("typeIDs", size)),
        positions(Coord3DView("positions", size)),
        forces(Coord3DView("forces", size)),
        oldForces(Coord3DView("oldForces", size)),
        velocities(Coord3DView("velocities", size)) {}

  void addParticle(const Particle &p) {
    if (size == capacity) {
      capacity *= 2;
      changes.rehash(capacity);
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
    Kokkos::parallel_for("addParticle", 1, KOKKOS_LAMBDA(int i) {
      typeIDs(size) = p.typeID;
      positions(size) = p.position;
      forces(size) = p.force;
      oldForces(size) = p.oldForce;
      velocities(size) = p.velocity;
    });
    ++size;
  }

  
//  void removeParticle(int index) {
//    --size;
//    Kokkos::parallel_for("removeParticle", 1, KOKKOS_LAMBDA(int i) {
//      typeIDs(index) = typeIDs(size);
//      positions(index) = positions(size);
//      forces(index) = forces(size);
//      velocities(index) = velocities(size);
//      oldForces(index) = oldForces(size);
//      if(changes.exists(index)) {
//        changes.erase(index);
//      }
//      if(changes.exists(size)) {
//        changes.insert(index, changes.valid_at(size));
//        changes.erase(size);
//      }
//    });
//    if(size < capacity / 4) {
//      capacity /= 2;
//      Kokkos::View<int *> newTypeIDs = Kokkos::View<int *>("typeIDs", capacity);
//      Coord3DView newPositions = Coord3DView("positions", capacity);
//      Coord3DView newForces = Coord3DView("forces", capacity);
//      Coord3DView newOldForces = Coord3DView("oldForces", capacity);
//      Coord3DView newVelocities = Coord3DView("velocities", capacity);
//      Kokkos::parallel_for(size, KOKKOS_LAMBDA(int i) {
//        newTypeIDs(i) = typeIDs(i);
//        newPositions(i) = positions(i);
//        newForces(i) = forces(i);
//        newOldForces(i) = oldForces(i);
//        newVelocities(i) = velocities(i);
//      });
//      typeIDs = newTypeIDs;
//      positions = newPositions;
//      forces = newForces;
//      oldForces = newOldForces;
//      velocities = newVelocities;
//    }
//  }
};
