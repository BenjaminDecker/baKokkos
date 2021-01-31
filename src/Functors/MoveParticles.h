//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"

class MoveParticles {
  const Kokkos::View<Coord3D**> positions;
  const Kokkos::View<Coord3D**> velocities;
  const Kokkos::View<Coord3D**> forces;
  const Kokkos::View<Coord3D**> oldForces;
  const Kokkos::View<int**> particleIDs;
  const Kokkos::View<int**> typeIDs;
  const Kokkos::View<int*> cellSizes;
  const Kokkos::View<bool*> hasMoved;
  const Kokkos::View<bool*> isHalo;
  const Kokkos::View<int> capacity;
  const Kokkos::View<bool> moveWasSuccessful;
  const Kokkos::View<int*> baseCells;
  const Coord3D boxMin;
  const int numCells[3];
  const int numCellsTotal;
  const double cutoff;
  const double deltaT;
  const BoundaryCondition boundaryCondition;

 public:
  MoveParticles(const Simulation &simulation, const Kokkos::View<int*> baseCells)
      : positions(simulation.positions),
        velocities(simulation.velocities),
        forces(simulation.forces),
        oldForces(simulation.oldForces),
        particleIDs(simulation.particleIDs),
        typeIDs(simulation.typeIDs),
        cellSizes(simulation.cellSizes),
        hasMoved(simulation.hasMoved),
        isHalo(simulation.isHalo),
        capacity(simulation.capacity),
        moveWasSuccessful(simulation.moveWasSuccessful),
        boxMin(simulation.boxMin),
        numCells{simulation.numCellsX, simulation.numCellsY, simulation.numCellsZ},
        numCellsTotal(simulation.numCells),
        cutoff(simulation.config.cutoff),
        deltaT(simulation.config.deltaT),
        boundaryCondition(simulation.boundaryCondition),
        baseCells(baseCells)
  {}

  [[nodiscard]] KOKKOS_INLINE_FUNCTION int getCorrectCellNumberDevice(const Coord3D &position) const {
    const Coord3D cellPosition = (position - boxMin) / cutoff;
    return static_cast<int>(cellPosition.z) * numCells[0] * numCells[1] +
        static_cast<int>(cellPosition.y) * numCells[0] +
        static_cast<int>(cellPosition.x);
  }

  KOKKOS_INLINE_FUNCTION
  void getRelativeCellCoordinatesDevice(int cellNumber, int &x, int &y, int &z) const {
    z = static_cast<int>(cellNumber / (numCells[0] * numCells[1]));
    cellNumber -= z * (numCells[0] * numCells[1]);
    y = static_cast<int>(cellNumber / numCells[0]);
    x = static_cast<int>(cellNumber - y * numCells[0]);
  }

  KOKKOS_INLINE_FUNCTION void operator() (int index) const {
    const int cellNumber = baseCells(index);
      if (!hasMoved(cellNumber)) {
        return;
      }
      for (int particleIndex = cellSizes(cellNumber) - 1; 0 <= particleIndex; --particleIndex) {
        Coord3D position = positions(cellNumber, particleIndex);
        const int targetCellNumber = getCorrectCellNumberDevice(position);
        if (cellNumber == targetCellNumber) {
          continue;
        }
        if (isHalo(targetCellNumber)) {
          switch (boundaryCondition) {
            case none: {
              --cellSizes(cellNumber);
              positions(cellNumber, particleIndex) = positions(cellNumber, cellSizes(cellNumber));
              velocities(cellNumber, particleIndex) = velocities(cellNumber, cellSizes(cellNumber));
              forces(cellNumber, particleIndex) = forces(cellNumber, cellSizes(cellNumber));
              oldForces(cellNumber, particleIndex) = oldForces(cellNumber, cellSizes(cellNumber));
              particleIDs(cellNumber, particleIndex) = particleIDs(cellNumber, cellSizes(cellNumber));
              typeIDs(cellNumber, particleIndex) = typeIDs(cellNumber, cellSizes(cellNumber));
            }
              break;
            case reflecting:
              // This should not happen, time step or particle energy was too large
              exit(-1);
            case periodic: {
              //TODO maybe there is a bug here
              int relativeX, relativeY, relativeZ;
              getRelativeCellCoordinatesDevice(targetCellNumber,relativeX,relativeY,relativeZ);
              const int correctX = relativeX;
              const int correctY = relativeY;
              const int correctZ = relativeZ;
              position += Coord3D(
                  (correctX == 0 ? 1 : correctX == numCells[0] - 1 ? -1 : 0) * cutoff * (numCells[0] - 2),
                  (correctY == 0 ? 1 : correctY == numCells[1] - 1 ? -1 : 0) * cutoff * (numCells[1] - 2),
                  (correctZ == 0 ? 1 : correctZ == numCells[2] - 1 ? -1 : 0) * cutoff * (numCells[2] - 2)
              );
              const auto otherCellNumber = getCorrectCellNumberDevice(position);
              auto &otherSize = cellSizes(otherCellNumber);
              if (otherSize == capacity()) {
                moveWasSuccessful() = false;
                return;
              }
              positions(otherCellNumber, otherSize) = positions(cellNumber, particleIndex);
              velocities(otherCellNumber, otherSize) = velocities(cellNumber, particleIndex);
              forces(otherCellNumber, otherSize) = forces(cellNumber, particleIndex);
              oldForces(otherCellNumber, otherSize) = oldForces(cellNumber, particleIndex);
              particleIDs(otherCellNumber, otherSize) = particleIDs(cellNumber, particleIndex);
              typeIDs(otherCellNumber, otherSize) = typeIDs(cellNumber, particleIndex);
              ++otherSize;

              auto &size = cellSizes(cellNumber);
              --size;
              positions(cellNumber, particleIndex) = positions(cellNumber, size);
              velocities(cellNumber, particleIndex) = velocities(cellNumber, size);
              forces(cellNumber, particleIndex) = forces(cellNumber, size);
              oldForces(cellNumber, particleIndex) = oldForces(cellNumber, size);
              particleIDs(cellNumber, particleIndex) = particleIDs(cellNumber, size);
              typeIDs(cellNumber, particleIndex) = typeIDs(cellNumber, size);
            }
              break;

          }
        } else {
          const auto cap = capacity();
          auto &targetSize = cellSizes(targetCellNumber);
          if (targetSize == capacity()) {
            moveWasSuccessful() = false;
            return;
          }
          positions(targetCellNumber, targetSize) = positions(cellNumber, particleIndex);
          velocities(targetCellNumber, targetSize) = velocities(cellNumber, particleIndex);
          forces(targetCellNumber, targetSize) = forces(cellNumber, particleIndex);
          oldForces(targetCellNumber, targetSize) = oldForces(cellNumber, particleIndex);
          particleIDs(targetCellNumber, targetSize) = particleIDs(cellNumber, particleIndex);
          typeIDs(targetCellNumber, targetSize) = typeIDs(cellNumber, particleIndex);
          ++targetSize;

          auto &size = cellSizes(cellNumber);
          --size;
          positions(cellNumber, particleIndex) = positions(cellNumber, size);
          velocities(cellNumber, particleIndex) = velocities(cellNumber, size);
          forces(cellNumber, particleIndex) = forces(cellNumber, size);
          oldForces(cellNumber, particleIndex) = oldForces(cellNumber, size);
          particleIDs(cellNumber, particleIndex) = particleIDs(cellNumber, size);
          typeIDs(cellNumber, particleIndex) = typeIDs(cellNumber, size);

        }
      }
      hasMoved(cellNumber) = false;
  }
};