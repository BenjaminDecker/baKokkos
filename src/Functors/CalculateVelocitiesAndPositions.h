//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"
#include "FunctorData.h"

/**
 * A functor for usage inside of a parallel_for(). For all particles in the cell with the specified cell number, this
 * calculates the new velocities of the particles based on the force acting on them and their previous velocities.
 * After this, the new positions based on the calculated velocities and the time step length is calculated.
 */
class CalculateVelocitiesAndPositions {

  /// A class that saves information about the simulation for the functors to use.
  const FunctorData data;

 public:
  explicit CalculateVelocitiesAndPositions(const FunctorData &functorData) : data(functorData) {}

  /**
   * Returns the cell number that corresponds to a particle position, this is used to calculate whether or not a
   * particle moved out of its cell.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION int getCorrectCellNumberDevice(const Coord3D &position) const {
    const Coord3D cellPosition = (position - data.boxMin) / data.cutoff;
    return static_cast<int>(cellPosition.z) * data.numCells[0] * data.numCells[1] +
        static_cast<int>(cellPosition.y) * data.numCells[0] +
        static_cast<int>(cellPosition.x);
  }

  /**
   * The operator of the functor. This function is called once for each index in a parallel_for().
   */
  KOKKOS_INLINE_FUNCTION void operator() (int cellNumber) const {

    // If the corresponding cell is a halo cell, all velocity and position calculations can be skipped.
    if(data.isHalo(cellNumber)) {
      return;
    }

    // All new velocities and positions are calculated
    for (int i = 0; i < data.cellSizes(cellNumber); ++i) {
      auto &velocity = data.velocities(cellNumber,i);
      auto &position = data.positions(cellNumber,i);
      const auto force = data.forces(cellNumber,i);
      const auto oldForce = data.oldForces(cellNumber,i);
      const auto mass = data.particleProperties.value_at(data.particleProperties.find(data.typeIDs(cellNumber,i))).mass;

      velocity += (force + oldForce) * (data.deltaT / (2 * mass));
      position += velocity * data.deltaT + force * ((data.deltaT * data.deltaT) / (2 * mass));

      const int correctCellNumber = getCorrectCellNumberDevice(position);

      // If a particle moved outside of the cell, the hasMoved variable is set
      if (cellNumber != correctCellNumber) {
        data.hasMoved(cellNumber) = true;
      }
    }
  }
};