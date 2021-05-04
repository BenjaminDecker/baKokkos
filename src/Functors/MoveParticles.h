//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"

/**
 * A functor for usage inside of a parallel_for(). For each cell with the specified cell number, this checks if any
 * particles left the cell during the last time step. If so, the functor tries to move these particles into their new cells.
 * The functor will fail if the capacity of the cells is to low to add another particle to a cell. In this case the cells
 * are resized outside of the functor and the functor is run again until there is enough capacity.
 */
class MoveParticles {

  /// A class that saves information about the simulation for the functors to use.
  const FunctorData data;

  /// A view that contains the cell numbers of the base steps that have to be calculated.
  const Kokkos::View<int*> baseCells;

 public:
  MoveParticles(const FunctorData &functorData, const Kokkos::View<int*> &baseCells) : data(functorData), baseCells(baseCells) {}


  /**
   * Returns the correct cell number a particle should be in. This is used to determine if a particle is inside the
   * correct cell.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION int correctCellNumber(const Coord3D &position) const {
    const Coord3D cellPosition = (position - data.boxMin) / data.cutoff;
    return static_cast<int>(cellPosition.z) * data.numCells[0] * data.numCells[1] +
        static_cast<int>(cellPosition.y) * data.numCells[0] +
        static_cast<int>(cellPosition.x);
  }

  /**
   * Writes the relative cell coordinates of the cell with the specified cell number into the specified variables.
   *
   * The relative cell coordinates of a cell are the coordinates into the grid of cells. The cell in the bottom front
   * left corner of the grid has the coordinates (0,0,0) and all other cells have coordinates (n,m,k) where n, m, k are
   * inside the natural numbers.
   * The relative cell coordinates are not the same as the bottom left corner coordinates of a cell.
   *
   * TODO maybe this can be moved into a super class of the functors to avoid code duplication. Have to test first, with
   * CUDA some things are not so simple...
   * TODO maybe I can use a std::array here, but I don't know if it works with CUDA
   */
  KOKKOS_INLINE_FUNCTION
  void relativeCellCoordinates(int cellNumber, int &x, int &y, int &z) const {
    z = cellNumber / (data.numCells[0] * data.numCells[1]);
    cellNumber -= z * (data.numCells[0] * data.numCells[1]);
    y = cellNumber / data.numCells[0];
    x = cellNumber - y * data.numCells[0];
  }

  /**
   * Removes the particle at the specified index from the cell with the specified index by copying the properties of
   * the last particle in this cell to the specified index and decrementing the cell size.
   */
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

  /**
   * Copies the particle at the specified index of the cell with the specified cell number into the cell with the
   * specified cell number, by copying its properties at the (last index + 1) of the other cell and incrementing its size.
   */
  KOKKOS_INLINE_FUNCTION bool copyParticle(int atIndex, int fromCell, int toCell) const {
    auto &targetCellSize = data.cellSizes(toCell);
    if (targetCellSize == data.capacity()) {
      return false;
    }
    data.positions(toCell, targetCellSize) = data.positions(fromCell, atIndex);
    data.velocities(toCell, targetCellSize) = data.velocities(fromCell, atIndex);
    data.forces(toCell, targetCellSize) = data.forces(fromCell, atIndex);
    data.oldForces(toCell, targetCellSize) = data.oldForces(fromCell, atIndex);
    data.particleIDs(toCell, targetCellSize) = data.particleIDs(fromCell, atIndex);
    data.typeIDs(toCell, targetCellSize) = data.typeIDs(fromCell, atIndex);
    ++targetCellSize;
    return true;
  }

    /**
   * The operator of the functor. This function is called once for each index in a parallel_for().
   */
  KOKKOS_INLINE_FUNCTION void operator() (int index) const {

    // baseCells contains all cell numbers for this base step. This saves the index of the next cell into cellNumber.
    const int cellNumber = baseCells(index);

    // If no particle has moved outside of this cell, the cell can be skipped.
      if (!data.hasMoved(cellNumber)) {
        return;
      }

      /*
       * The iteration over the particles of the cell happens in reverse order, so when a particle is moved, the index
       * does not have to be adapted.
       * For example, if you iterate normally and the particle at index 2 is removed, the next particle would be at
       * index 2 now, not at index 3. Iterating in reverse order prevents this issue.
       */
      for (int particleIndex = data.cellSizes(cellNumber) - 1; 0 <= particleIndex; --particleIndex) {
        Coord3D position = data.positions(cellNumber, particleIndex);

        // If the number of this cell and the cell number a particle should be in match, nothing has to be done.
        const int targetCellNumber = correctCellNumber(position);
        if (cellNumber == targetCellNumber) {
          continue;
        }

        // The particle has to be moved.

        // If the target cell is a halo cell, what happens to the particle depends on the boundary condition used.
        if (data.isHalo(targetCellNumber)) {
          switch (data.boundaryCondition) {

            // The particle is just removed from this cell. It does not exist anymore.
            case none: {
              removeParticle(particleIndex, cellNumber);
            }
              break;

              /*
               * If a particle moved into a halo cell with reflecting boundary conditions, something went wrong. Most
               * likely the time step is too long or the particle speed is too high.
               */
            case reflecting:
              exit(42);

              /*
               * With periodic boundary conditions, the particle has to be "teleported" to the corresponding cell on the
               * other side of the simulation instead of entering the halo cell.
               */
            case periodic: {
              //TODO maybe there is a bug here, I take a look at it later
              int relativeX, relativeY, relativeZ;
              relativeCellCoordinates(targetCellNumber, relativeX, relativeY, relativeZ);

              /*
               * Depending on the side of the simulation cube the halo cell is on, the particle has to be moved by a
               * total of [cutoff * ([total cells in this direction] - 2)] in the corresponding direction to be inside
               * the correct cell.
               */
              position += Coord3D(
                  (relativeX == 0 ? 1 : relativeX == data.numCells[0] - 1 ? -1 : 0) * data.cutoff * (data.numCells[0] - 2),
                  (relativeY == 0 ? 1 : relativeY == data.numCells[1] - 1 ? -1 : 0) * data.cutoff * (data.numCells[1] - 2),
                  (relativeZ == 0 ? 1 : relativeZ == data.numCells[2] - 1 ? -1 : 0) * data.cutoff * (data.numCells[2] - 2)
              );

              const auto otherCellNumber = correctCellNumber(position);

              /*
               * After the particle position is updated, the particle information has to be moved into the new cell. If
               * the move was not successful, the capacity of the cells was to low and the functor returns. The cells
               * are then resized from host space.
               */
              if(!copyParticle(particleIndex, cellNumber, otherCellNumber)) {
                data.moveWasSuccessful() = false;
                return;
              }

              // If everything worked, the particle was copied successfully and can be deleted from this cell.
              removeParticle(particleIndex, cellNumber);
            }
              break;
          }

          // If the target cell is not a halo cell the particle is just moved from this cell into the target cell
        } else {

          /*
           * After the particle position is updated, the particle information has to be moved into the new cell. If
           * the move was not successful, the capacity of the cells was to low and the functor returns. The cells
           * are then resized from host space.
           */
          if (!copyParticle(particleIndex, cellNumber, targetCellNumber)) {
            data.moveWasSuccessful() = false;
            return;
          }
          // If everything worked, the particle was copied successfully and can be deleted from this cell.
          removeParticle(particleIndex, cellNumber);
        }
      }
      // After all particles were checked, the hasMoved property can be set to false again.
    data.hasMoved(cellNumber) = false;
  }
};