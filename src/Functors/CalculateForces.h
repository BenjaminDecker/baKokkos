//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"
#include "FunctorData.h"

/// Epsilon for particle interaction in the lennard-jones potential. This value has to be added to the ParticleProperties map
constexpr float epsilon = 1;

/// Sigma for particle interaction in the lennard-jones potential. This value has to be added to the ParticleProperties map
constexpr float sigma = 1;

/// The pre-computed value of sigma^6
constexpr float sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;

/// The pre-computed value of 24 * epsilon * sigma^6
constexpr float twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;

/// The pre-computed value of 2 * (24 * epsilon * sigma^6) * sigma^6
constexpr float fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

/**
 * A functor for usage inside of a parallel_for(). Calculates the forces between particles for the c08 base step with
 * the specified cell number.
 */
class CalculateForces {

  /// A class that saves information about the simulation for the functors to use.
  const FunctorData data;

  /// A view that contains the cell numbers of the base steps that have to be calculated.
  const Kokkos::View<int*> baseCells;

 public:

  explicit CalculateForces(const FunctorData &functorData, const Kokkos::View<int*> &baseCells) : data(functorData), baseCells(baseCells) {}

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
   * The calculator function for the force calculation. Returns the force that two particles exert on another based on
   * their distance to another. The values for sigma and epsilon are still hard coded, but later be loaded from the
   * ParticleProperties map.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D calculator(const Coord3D &distance) const {
    const float distanceValue = distance.absoluteValue();
    if (distanceValue > data.cutoff) {
      return Coord3D();
    }
    const float distanceValuePow6 =
        distanceValue * distanceValue * distanceValue * distanceValue * distanceValue *
            distanceValue;
    const float distanceValuePow13 = distanceValuePow6 * distanceValuePow6 * distanceValue;

    // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
    const float forceValue =
        (twentyFourEpsilonSigmaPow6 * distanceValuePow6 - fourtyEightEpsilonSigmaPow12) /
            distanceValuePow13;
    return (distance * (forceValue / distanceValue));
  }

  /**
   * The operator of the functor. This function is called once for each index in a parallel_for().
   */
  KOKKOS_INLINE_FUNCTION void operator() (int index) const {
    const int baseCellNumber = baseCells(index);
    int relativeX, relativeY, relativeZ;
    relativeCellCoordinates(baseCellNumber,
                            relativeX,
                            relativeY,
                            relativeZ);

    /*
     * This tests if the base cell is somewhere along the back top right sides of the simulation cube. If so, the c08
     * base step is skipped. This could later be pre calculated and saved inside a view for a slightly faster check.
     */
    if (relativeX == data.numCells[0] - 1 ||
        relativeY == data.numCells[1] - 1 ||
        relativeZ == data.numCells[2] - 1) {
      return;
    }

    /*
     * For each particle in the base cell with the index of the current thread, the force calculation with all other
     * particles in the same base cell are calculated.
     */
    for (int id_1 = 0; id_1 < data.cellSizes(baseCellNumber); ++id_1) {
      for (int id_2 = id_1 + 1; id_2 < data.cellSizes(baseCellNumber); ++id_2) {

        const Coord3D actingForce = calculator(data.positions(baseCellNumber,id_1).distanceTo(data.positions(baseCellNumber,id_2)));
        data.forces(baseCellNumber,id_1) += actingForce;
        data.forces(baseCellNumber,id_2) += actingForce * (-1);
      }
    }

    // For each pair of the 13 pairs in a c08 base step
    for (int pairNumber = 0; pairNumber < 13; ++pairNumber) {

      // Get the two cells corresponding to the current pair
      const int cellOneNumber = data.c08Pairs(baseCellNumber, pairNumber, 0);
      const int cellTwoNumber = data.c08Pairs(baseCellNumber, pairNumber, 1);

      // If both of the cells are non-halo cells
      if (!data.isHalo(cellOneNumber) && !data.isHalo(cellTwoNumber)) {

        // For each particle pair with one particle from each cell
        for (int id_1 = 0; id_1 < data.cellSizes(cellOneNumber); ++id_1) {
          for (int id_2 = 0; id_2 < data.cellSizes(cellTwoNumber); ++id_2) {

            // Calculate the force once and write it to both particles with different signs
            const Coord3D actingForce = calculator(data.positions(cellOneNumber,id_1).distanceTo(data.positions(cellTwoNumber,id_2)));
            data.forces(cellOneNumber,id_1) += actingForce;
            data.forces(cellTwoNumber,id_2) += actingForce * (-1);
          }
        }
      }
      // If at least one of the cells is a halo cell
      else {

        // If both cells are halo cells, continue. After this, only cell pairs with at least one halo cell remain.
        if (data.isHalo(cellOneNumber) && data.isHalo(cellTwoNumber)) {
          continue;
        }

        // Set normalCellNumber to the cell number of the non-halo cell and haloCellNumber to the other
        const int normalCellNumber = data.isHalo(cellOneNumber) ? cellTwoNumber : cellOneNumber;
        const int haloCellNumber = data.isHalo(cellOneNumber) ? cellOneNumber : cellTwoNumber;

        // Switch based on the boundary condition that is used in the simulation
        switch (data.boundaryCondition) {

          /*
           * If no boundary condition is used, nothing should be done for particles inside halo cells. There should not
           * be any particles inside these cells anyways.
           */
          case none:break;

          /*
           * With periodic boundary conditions, particles on the edge of the simulation should interact with particles on
           * the opposite side of the simulation.
           */
          case periodic: {
            // Find the cell number of the cell on the other side of the simulation that the halo cell corresponds to
            const int periodicTargetCellNumber = data.periodicTargetCellNumbers(haloCellNumber);

            // Calculate the distance from the periodic target cell to the halo cell.
            const Coord3D offset =
                data.bottomLeftCorners(periodicTargetCellNumber).distanceTo(data.bottomLeftCorners(haloCellNumber));

            /*
             * For each particle pair with one particle from each cell, calculate the forces that these particles would
             * exert on another if the particles from the periodic target cell were inside of the halo cell instead.
             */
            //TODO Maybe a bug? Why is the force only written to one of the particles from the pair? I think there was a reason, but I cant remember.
            for (int id_1 = 0; id_1 < data.cellSizes(normalCellNumber); ++id_1) {
              for (int id_2 = 0; id_2 < data.cellSizes(periodicTargetCellNumber); ++id_2) {
                data.forces(normalCellNumber,id_1) += calculator(data.positions(normalCellNumber,id_1).distanceTo(data.positions(periodicTargetCellNumber,id_2)) + offset);
              }
            }
            break;
          }

          // With reflecting boundary conditions, reflections with the halo cells should be simulated with ghost particles
          case reflecting: {

            // Calculate the distance from the normal cell to the halo cell.
            const Coord3D offset =
                data.bottomLeftCorners(normalCellNumber).distanceTo(data.bottomLeftCorners(haloCellNumber));

            /*
             * Continue if the current halo cell does not share a side with the normal cell, but instead only shares a
             * corner of edge. Ghost particles are like reflections with the cell wall between the two cells. There are no
             * ghost particles in halo cells that are diagonally from the normal cell.
             */
            if (std::abs(offset.x) + std::abs(offset.y) + std::abs(offset.z) > data.cutoff) {
              continue;
            }

            // For each particle in the normal cell
            for (int id = 0; id < data.cellSizes(normalCellNumber); ++id) {
              const Coord3D position = data.positions(normalCellNumber,id);

              // Create a ghost position in the halo cell with the same in-cell offset as the normal particle position
              Coord3D ghostPosition = position + offset;
              const auto haloCellBottomLeftCorner = data.bottomLeftCorners(haloCellNumber);

              /*
               * Test the direction that the new position was shifted, and reflect the ghost position in the corresponding
               * direction accodingly. Only one of the spacial directions can be != 0, because only cell pairs that
               * share a side remain at this point.
               */
              if (offset.x != 0) {
                ghostPosition.x = haloCellBottomLeftCorner.x
                    + (data.cutoff - (ghostPosition.x - haloCellBottomLeftCorner.x));
              } else if (offset.y != 0) {
                ghostPosition.y = haloCellBottomLeftCorner.y
                    + (data.cutoff - (ghostPosition.y - haloCellBottomLeftCorner.y));
              } else if (offset.z != 0) {
                ghostPosition.z = haloCellBottomLeftCorner.z
                    + (data.cutoff - (ghostPosition.z - haloCellBottomLeftCorner.z));
              }

              // Add the force from the ghost particle to the normal particle.
              data.forces(normalCellNumber,id) += calculator(position.distanceTo(ghostPosition));
            }
            break;
          }
        }
      }
    }
  }
};
