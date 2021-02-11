//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"

class CalculateVelocitiesAndPositions {
  const Kokkos::View<Coord3D**> positions;
  const Kokkos::View<Coord3D**> velocities;
  const Kokkos::View<Coord3D**> forces;
  const Kokkos::View<Coord3D**> oldForces;
  const Kokkos::View<int**> typeIDs;
  const Kokkos::View<int*> cellSizes;
  const Kokkos::View<bool*> hasMoved;
  const Kokkos::UnorderedMap<int, ParticleProperties> particleProperties;
  const Coord3D boxMin;
  const int numCells[3];
  const double cutoff;
  const double deltaT;
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

  [[nodiscard]] KOKKOS_INLINE_FUNCTION int getCorrectCellNumberDevice(const Coord3D &position) const {
    const Coord3D cellPosition = (position - boxMin) / cutoff;
    return static_cast<int>(cellPosition.z) * numCells[0] * numCells[1] +
        static_cast<int>(cellPosition.y) * numCells[0] +
        static_cast<int>(cellPosition.x);
  }

  KOKKOS_INLINE_FUNCTION void operator() (int cellNumber) const {
    if(isHalo(cellNumber)) {
      return;
    }
    for (int i = 0; i < cellSizes(cellNumber); ++i) {
      auto &velocity = velocities(cellNumber,i);
      auto &position = positions(cellNumber,i);
      const auto force = forces(cellNumber,i);
      const auto oldForce = oldForces(cellNumber,i);
      const auto mass = particleProperties.value_at(particleProperties.find(typeIDs(cellNumber,i))).mass;

      velocity += (force + oldForce) * (deltaT / (2 * mass));
      position += velocity * deltaT + force * ((deltaT * deltaT) / (2 * mass));

      const int correctCellNumber = getCorrectCellNumberDevice(position);
      if (cellNumber != correctCellNumber) {
        hasMoved(cellNumber) = true;
      }
    }
  }
};