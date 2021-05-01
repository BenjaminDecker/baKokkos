//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"

/**
 * A functor for use inside of a parallel_for(). For all particles in the cell with the specified cell number, this
 * calculates the new velocities of the particles based on the force acting on them and their previous velocities.
 * After this, the new positions based on the calculated velocities and the time step length is calculated.
 */
class CalculateVelocitiesAndPositions {
  /**
   * A 2-dimensional view that saves all particle positions in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  const Kokkos::View<Coord3D**> positions;

  /**
   * A 2-dimensional view that saves all particle velocities in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  const Kokkos::View<Coord3D**> velocities;

  /**
   * A 2-dimensional view that saves all particle forces in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  const Kokkos::View<Coord3D**> forces;

  /**
   * A 2-dimensional view that saves all old particle forces in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  const Kokkos::View<Coord3D**> oldForces;

  /**
   * A 2-dimensional view that saves all particle typeIDs in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  const Kokkos::View<int**> typeIDs;

  /// A view that saves the amount of particles inside each cell.
  const Kokkos::View<int*> cellSizes;

  /// A view that saves whether or not a particle has moved outside of a cell in the last time step.
  const Kokkos::View<bool*> hasMoved;

  /// A mapping from particle IDs to various particle properties.
  const Kokkos::UnorderedMap<int, ParticleProperties> particleProperties;

  /**
   * The coordinate of the bottom left front corner of the simulation cube. Together with the numCells variable, this
   * is used to calculate whether or not a particle moved out of its cell.
   */
  const Coord3D boxMin;

  /**
   * The amount of cells in each spacial direction in the simulation. Together with the boxMin variable, this
   * is used to calculate whether or not a particle moved out of its cell.
   */
  const int numCells[3];

  /// Maximum distance between two particles for which the force calculation can not be neglected to increase performance
  const float cutoff;

  /// Length of one time step of the simulation
  const float deltaT;

  /// A view that saves for each cell whether it is a halo cell or not.
  const Kokkos::View<bool*> isHalo;

 public:
  explicit CalculateVelocitiesAndPositions(const Simulation &simulation)
      : positions(simulation.positions),
        velocities(simulation.velocities),
        forces(simulation.forces),
        oldForces(simulation.oldForces),
        typeIDs(simulation.typeIDs),
        cellSizes(simulation.cellSizes),
        hasMoved(simulation.hasMoved),
        particleProperties(simulation.particleProperties),
        boxMin(simulation.boxMin),
        numCells{simulation.numCellsX, simulation.numCellsY, simulation.numCellsZ},
        cutoff(simulation.config.cutoff),
        deltaT(simulation.config.deltaT),
        isHalo(simulation.isHalo){}

  /**
   * Returns the cell number that corresponds to a particle position, this is used to calculate whether or not a
   * particle moved out of its cell.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION int getCorrectCellNumberDevice(const Coord3D &position) const {
    const Coord3D cellPosition = (position - boxMin) / cutoff;
    return static_cast<int>(cellPosition.z) * numCells[0] * numCells[1] +
        static_cast<int>(cellPosition.y) * numCells[0] +
        static_cast<int>(cellPosition.x);
  }

  /**
   * The operator of the functor. This function is called once for each index in a parallel_for().
   */
  KOKKOS_INLINE_FUNCTION void operator() (int cellNumber) const {

    // If the corresponding cell is a halo cell, all velocity and position calculations can be skipped.
    if(isHalo(cellNumber)) {
      return;
    }

    // All new velocities and positions are calculated
    for (int i = 0; i < cellSizes(cellNumber); ++i) {
      auto &velocity = velocities(cellNumber,i);
      auto &position = positions(cellNumber,i);
      const auto force = forces(cellNumber,i);
      const auto oldForce = oldForces(cellNumber,i);
      const auto mass = particleProperties.value_at(particleProperties.find(typeIDs(cellNumber,i))).mass;

      velocity += (force + oldForce) * (deltaT / (2 * mass));
      position += velocity * deltaT + force * ((deltaT * deltaT) / (2 * mass));

      const int correctCellNumber = getCorrectCellNumberDevice(position);

      // If a particle moved outside of the cell, the hasMoved variable is set
      if (cellNumber != correctCellNumber) {
        hasMoved(cellNumber) = true;
      }
    }
  }
};