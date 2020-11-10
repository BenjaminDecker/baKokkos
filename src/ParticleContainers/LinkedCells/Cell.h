//
// Created by Benjamin Decker on 10/11/2020.
//

#pragma once

#include <Kokkos_UnorderedMap.hpp>
#include "LinkedCellsParticleContainer.h"

struct Cell {
  int size;
  int capacity;
  Kokkos::UnorderedMap<int, int>
      changes; /**< Saves if particles left this cell during the last iteration and their new cell number */
  Kokkos::View<int *> typeIDs; /**< Type identifiers for looking up further particle properties */
  Coord3DView positions; /**< Array of 3-dimensional position vectors */
  Coord3DView forces; /**< Array of 3-dimensional force vectors acting on particles */
  Coord3DView oldForces; /**< Array of 3-dimensional force vectors acting on particles from the previous iteration */
  Coord3DView velocities; /**< Array of 3-dimensional velocity vectors */

  explicit Cell(int capacity)
      : size(0),
        capacity(capacity),
        changes(Kokkos::UnorderedMap<int, int>(capacity)),
        typeIDs(Kokkos::View<int *>("typeIDs", size)),
        positions(Coord3DView("positions", size)),
        forces(Coord3DView("forces", size)),
        oldForces(Coord3DView("oldForces", size)),
        velocities(Coord3DView("velocities", size)) {}

};
