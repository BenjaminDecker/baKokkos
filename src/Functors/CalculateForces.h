//
// Created by Benjamin Decker on 31/01/2021.
//

#pragma once

#include <Kokkos_Core.hpp>
#include "../Simulation.h"

constexpr double epsilon = 1;
constexpr double sigma = 1;
constexpr double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
constexpr double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
constexpr double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

class CalculateForces {
  const Kokkos::View<Coord3D**> positions;
  const Kokkos::View<Coord3D**> forces;
  const Kokkos::View<int*> cellSizes;
  const Kokkos::View<int *[13][2]> c08Pairs;
  const Kokkos::View<int *> periodicTargetCellNumbers;
  const Kokkos::View<bool*> isHalo;
  const Kokkos::View<Coord3D *> bottomLeftCorners;
  const Kokkos::View<int*> baseCells;
  const int numCells[3];
  const double cutoff;
  const BoundaryCondition boundaryCondition;

 public:
  CalculateForces(const Simulation &simulation, const Kokkos::View<int*> baseCells)
      : positions(simulation.positions),
        forces(simulation.forces),
        cellSizes(simulation.cellSizes) ,
        c08Pairs(simulation.c08Pairs),
        periodicTargetCellNumbers(simulation.periodicTargetCellNumbers),
        isHalo(simulation.isHalo),
        bottomLeftCorners(simulation.bottomLeftCorners),
        numCells{simulation.numCellsX, simulation.numCellsY, simulation.numCellsZ},
        cutoff(simulation.config.cutoff),
        boundaryCondition(simulation.boundaryCondition),
        baseCells(baseCells)
  {}

  KOKKOS_INLINE_FUNCTION
  void getRelativeCellCoordinatesDevice(int cellNumber, int &x, int &y, int &z) const {
    z = cellNumber / (numCells[0] * numCells[1]);
    cellNumber -= z * (numCells[0] * numCells[1]);
    y = cellNumber / numCells[0];
    x = cellNumber - y * numCells[0];
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D calculator(const Coord3D &distance) const {
    const double distanceValue = distance.absoluteValue();
    if (distanceValue > cutoff) {
      return Coord3D();
    }
    const double distanceValuePow6 =
        distanceValue * distanceValue * distanceValue * distanceValue * distanceValue *
            distanceValue;
    const double distanceValuePow13 = distanceValuePow6 * distanceValuePow6 * distanceValue;

    // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
    const double forceValue =
        (twentyFourEpsilonSigmaPow6 * distanceValuePow6 - fourtyEightEpsilonSigmaPow12) /
            distanceValuePow13;
    return (distance * (forceValue / distanceValue));
  }

  KOKKOS_INLINE_FUNCTION void operator() (int index) const {
    const int baseCellNumber = baseCells(index);
    int relativeX, relativeY, relativeZ;
    getRelativeCellCoordinatesDevice(baseCellNumber,
                                     relativeX,
                                     relativeY,
                                     relativeZ);
    // This can be done in initializeSimulation()
    if (relativeX == numCells[0] - 1 ||
        relativeY == numCells[1] - 1 ||
        relativeZ == numCells[2] - 1) {
      return;
    }
    for (int id_1 = 0; id_1 < cellSizes(baseCellNumber); ++id_1) {
      for (int id_2 = id_1 + 1; id_2 < cellSizes(baseCellNumber); ++id_2) {

        const Coord3D actingForce = calculator(positions(baseCellNumber,id_1).distanceTo(positions(baseCellNumber,id_2)));
        forces(baseCellNumber,id_1) += actingForce;
        forces(baseCellNumber,id_2) += actingForce * (-1);
      }
    }
    for (int pairNumber = 0; pairNumber < 13; ++pairNumber) {
      const int cellOneNumber = c08Pairs(baseCellNumber, pairNumber, 0);
      const int cellTwoNumber = c08Pairs(baseCellNumber, pairNumber, 1);
      if (!isHalo(cellOneNumber) && !isHalo(cellTwoNumber)) {
        for (int id_1 = 0; id_1 < cellSizes(cellOneNumber); ++id_1) {
          for (int id_2 = 0; id_2 < cellSizes(cellTwoNumber); ++id_2) {
            const Coord3D actingForce = calculator(positions(cellOneNumber,id_1).distanceTo(positions(cellTwoNumber,id_2)));
            forces(cellOneNumber,id_1) += actingForce;
            forces(cellTwoNumber,id_2) += actingForce * (-1);
          }
        }
      } else {
        if (isHalo(cellOneNumber) && isHalo(cellTwoNumber)) {
          continue;
        }
        const int normalCellNumber = isHalo(cellOneNumber) ? cellTwoNumber : cellOneNumber;
        const int haloCellNumber = isHalo(cellOneNumber) ? cellOneNumber : cellTwoNumber;
        switch (boundaryCondition) {
          case none:break;
          case periodic: {
            const int periodicTargetCellNumber = periodicTargetCellNumbers(haloCellNumber);
            const Coord3D offset =
                bottomLeftCorners(periodicTargetCellNumber).distanceTo(bottomLeftCorners(haloCellNumber));
            for (int id_1 = 0; id_1 < cellSizes(normalCellNumber); ++id_1) {
              for (int id_2 = 0; id_2 < cellSizes(periodicTargetCellNumber); ++id_2) {
                forces(normalCellNumber,id_1) += calculator(positions(normalCellNumber,id_1).distanceTo(positions(periodicTargetCellNumber,id_2)) + offset);
              }
            }
            break;
          }
          case reflecting: {
            const Coord3D offset =
                bottomLeftCorners(normalCellNumber).distanceTo(bottomLeftCorners(haloCellNumber));
            if (std::abs(offset.x) + std::abs(offset.y) + std::abs(offset.z) > cutoff) {
              continue;
            }
            for (int id = 0; id < cellSizes(normalCellNumber); ++id) {
              const Coord3D position = positions(normalCellNumber,id);
              Coord3D ghostPosition = position + offset;
              const auto haloCellBottomLeftCorner = bottomLeftCorners(haloCellNumber);
              if (offset.x != 0) {
                ghostPosition.x = haloCellBottomLeftCorner.x
                    + (cutoff - (ghostPosition.x - haloCellBottomLeftCorner.x));
              } else if (offset.y != 0) {
                ghostPosition.y = haloCellBottomLeftCorner.y
                    + (cutoff - (ghostPosition.y - haloCellBottomLeftCorner.y));
              } else if (offset.z != 0) {
                ghostPosition.z = haloCellBottomLeftCorner.z
                    + (cutoff - (ghostPosition.z - haloCellBottomLeftCorner.z));
              }
              forces(normalCellNumber,id) += calculator(position.distanceTo(ghostPosition));
            }
            break;
          }
        }
      }
    }
  }
};
