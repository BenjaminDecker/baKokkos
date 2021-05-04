//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"

class MoveParticles {

  const FunctorData data;

  /// A view that contains the cell numbers of the base steps that have to be calculated.
  const Kokkos::View<int*> baseCells;

 public:
  MoveParticles(const FunctorData &functorData, const Kokkos::View<int*> &baseCells) : data(functorData), baseCells(baseCells) {}

  [[nodiscard]] KOKKOS_INLINE_FUNCTION int getCorrectCellNumberDevice(const Coord3D &position) const {
    const Coord3D cellPosition = (position - data.boxMin) / data.cutoff;
    return static_cast<int>(cellPosition.z) * data.numCells[0] * data.numCells[1] +
        static_cast<int>(cellPosition.y) * data.numCells[0] +
        static_cast<int>(cellPosition.x);
  }

  KOKKOS_INLINE_FUNCTION
  void getRelativeCellCoordinatesDevice(int cellNumber, int &x, int &y, int &z) const {
    z = static_cast<int>(cellNumber / (data.numCells[0] * data.numCells[1]));
    cellNumber -= z * (data.numCells[0] * data.numCells[1]);
    y = static_cast<int>(cellNumber / data.numCells[0]);
    x = static_cast<int>(cellNumber - y * data.numCells[0]);
  }

  KOKKOS_INLINE_FUNCTION void removeParticle(int index, int cellNumber) const {
    auto &size = data.cellSizes(cellNumber);
    --size;
    data.positions(cellNumber, index) = data.positions(cellNumber, size);
    data.velocities(cellNumber, index) = data.velocities(cellNumber, size);
    data.forces(cellNumber, index) = data.forces(cellNumber, size);
    data.oldForces(cellNumber, index) = data.oldForces(cellNumber, size);
    data.particleIDs(cellNumber, index) = data.particleIDs(cellNumber, size);
    data.typeIDs(cellNumber, index) = data.typeIDs(cellNumber, size);
  }

  KOKKOS_INLINE_FUNCTION void operator() (int index) const {
    const int cellNumber = baseCells(index);
      if (!data.hasMoved(cellNumber)) {
        return;
      }
      for (int particleIndex = data.cellSizes(cellNumber) - 1; 0 <= particleIndex; --particleIndex) {
        Coord3D position = data.positions(cellNumber, particleIndex);
        const int targetCellNumber = getCorrectCellNumberDevice(position);
        if (cellNumber == targetCellNumber) {
          continue;
        }
        if (data.isHalo(targetCellNumber)) {
          switch (data.boundaryCondition) {
            case none: {
              removeParticle(particleIndex, cellNumber);
            }
              break;
            case reflecting:
              // This should not happen, time step or particle energy was too large
              exit(42);
            case periodic: {
              //TODO maybe there is a bug here
              int relativeX, relativeY, relativeZ;
              getRelativeCellCoordinatesDevice(targetCellNumber,relativeX,relativeY,relativeZ);
              const int correctX = relativeX;
              const int correctY = relativeY;
              const int correctZ = relativeZ;
              position += Coord3D(
                  (correctX == 0 ? 1 : correctX == data.numCells[0] - 1 ? -1 : 0) * data.cutoff * (data.numCells[0] - 2),
                  (correctY == 0 ? 1 : correctY == data.numCells[1] - 1 ? -1 : 0) * data.cutoff * (data.numCells[1] - 2),
                  (correctZ == 0 ? 1 : correctZ == data.numCells[2] - 1 ? -1 : 0) * data.cutoff * (data.numCells[2] - 2)
              );
              const auto otherCellNumber = getCorrectCellNumberDevice(position);
              auto &otherSize = data.cellSizes(otherCellNumber);
              if (otherSize == data.capacity()) {
                data.moveWasSuccessful() = false;
                return;
              }
              data.positions(otherCellNumber, otherSize) = data.positions(cellNumber, particleIndex);
              data.velocities(otherCellNumber, otherSize) = data.velocities(cellNumber, particleIndex);
              data.forces(otherCellNumber, otherSize) = data.forces(cellNumber, particleIndex);
              data.oldForces(otherCellNumber, otherSize) = data.oldForces(cellNumber, particleIndex);
              data.particleIDs(otherCellNumber, otherSize) = data.particleIDs(cellNumber, particleIndex);
              data.typeIDs(otherCellNumber, otherSize) = data.typeIDs(cellNumber, particleIndex);
              ++otherSize;

              removeParticle(particleIndex, cellNumber);
            }
              break;

          }
        } else {
          const auto cap = data.capacity();
          auto &targetSize = data.cellSizes(targetCellNumber);
          if (targetSize == data.capacity()) {
            data.moveWasSuccessful() = false;
            return;
          }
          data.positions(targetCellNumber, targetSize) = data.positions(cellNumber, particleIndex);
          data.velocities(targetCellNumber, targetSize) = data.velocities(cellNumber, particleIndex);
          data.forces(targetCellNumber, targetSize) = data.forces(cellNumber, particleIndex);
          data.oldForces(targetCellNumber, targetSize) = data.oldForces(cellNumber, particleIndex);
          data.particleIDs(targetCellNumber, targetSize) = data.particleIDs(cellNumber, particleIndex);
          data.typeIDs(targetCellNumber, targetSize) = data.typeIDs(cellNumber, particleIndex);
          ++targetSize;

          removeParticle(particleIndex, cellNumber);

        }
      }
    data.hasMoved(cellNumber) = false;
  }
};