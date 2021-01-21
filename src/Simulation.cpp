//
// Created by Benjamin Decker on 29.12.20.
//

#include "Simulation.h"
#include <iomanip>
#include <fstream>
#include <utility>

constexpr enum BoundaryCondition {
  none, periodic, reflecting
} boundaryCondition(reflecting);

KOKKOS_INLINE_FUNCTION
void getRelativeCellCoordinatesDevice(int cellNumber, int cellsX, int cellsY, int &x, int &y, int &z) {
  z = cellNumber / (cellsX * cellsY);
  cellNumber -= z * (cellsX * cellsY);
  y = cellNumber / cellsX;
  x = cellNumber - y * cellsX;
}

KOKKOS_INLINE_FUNCTION
int getCellNumberFromRelativeCellCoordinatesDevice(int x, int y, int z, int numCellsX, int numCellsY) {
  return z * numCellsX * numCellsY + y * numCellsX + x;
}

KOKKOS_INLINE_FUNCTION
int getCorrectCellNumberDevice(const Coord3D &position,
                               const Coord3D &boxMin,
                               double cutoff,
                               int numCellsX,
                               int numCellsY) {
  const Coord3D cellPosition = (position - boxMin) / cutoff;
  return getCellNumberFromRelativeCellCoordinatesDevice(static_cast<int>(cellPosition.x),
                                                        static_cast<int>(cellPosition.y),
                                                        static_cast<int>(cellPosition.z),
                                                        numCellsX,
                                                        numCellsY);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION
Coord3D calculator(const Coord3D &distance, double cutoff) {
  constexpr double epsilon = 1;
  constexpr double sigma = 1;
  constexpr double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  constexpr double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  constexpr double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;
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

Simulation::Simulation(SimulationConfig config) : config(std::move(config)), iteration(0) {
  initializeSimulation();
  cellSizes.sync_device();
}

void Simulation::start() {
  spdlog::info("Running Simulation...");
  Kokkos::Timer timer;

  //Iteration loop
  for (; iteration < 50; ++iteration) {
    if (true) {
      spdlog::info("Iteration: {:0" + std::to_string(std::to_string(config.iterations).length()) + "d}", iteration);
    }
    Kokkos::Profiling::pushRegion("Iteration: " + std::to_string(iteration));
    spdlog::info("forces");
    calculateForcesNewton3();
    spdlog::info("velocitiesAndPositions");
    calculateVelocitiesAndPositions();
    spdlog::info("move");
    moveParticles();
    spdlog::info("write");
//    if (config.vtk && iteration % config.vtk.value().second == 0) {
//      writeVTKFile(config.vtk.value().first);
//    }
    Kokkos::Profiling::popRegion();
  }

  const double time = timer.seconds();
  spdlog::info("Finished simulating. Time: " + std::to_string(time) + " seconds.");
}

void Simulation::addParticle(const Particle &particle) {
  const int cellNumber = getCorrectCellNumber(particle);
  if (cellNumber < 0 || numCells <= cellNumber) {
    std::cout
        << "Particles outside of the simulation space cuboid cannot be added to any cell of the simulation."
        << std::endl;
    exit(1);
  }

  int &size = cellSizes.view_host()(cellNumber);
  if (size == capacities) {
    capacities *= 2;
    Kokkos::resize(positions, numCells, capacities);
    Kokkos::resize(velocities, numCells, capacities);
    Kokkos::resize(forces, numCells, capacities);
    Kokkos::resize(oldForces, numCells, capacities);
    Kokkos::resize(particleIDs, numCells, capacities);
    Kokkos::resize(typeIDs, numCells, capacities);
  }
  Kokkos::deep_copy(Kokkos::subview(positions, cellNumber, size), particle.position);
  Kokkos::deep_copy(Kokkos::subview(velocities, cellNumber, size), particle.velocity);
  Kokkos::deep_copy(Kokkos::subview(forces, cellNumber, size), particle.force);
  Kokkos::deep_copy(Kokkos::subview(oldForces, cellNumber, size), particle.oldForce);
  Kokkos::deep_copy(Kokkos::subview(particleIDs, cellNumber, size), particle.particleID);
  Kokkos::deep_copy(Kokkos::subview(typeIDs, cellNumber, size), particle.typeID);
  ++size;
  cellSizes.modify_host();
}

std::vector<Particle> Simulation::getParticles(int cellNumber) const {
  std::vector<Particle> particles;

  auto h_positions = Kokkos::create_mirror_view(Kokkos::subview(positions, cellNumber, Kokkos::ALL));
  auto h_velocities = Kokkos::create_mirror_view(Kokkos::subview(velocities, cellNumber, Kokkos::ALL));
  auto h_forces = Kokkos::create_mirror_view(Kokkos::subview(forces, cellNumber, Kokkos::ALL));
  auto h_oldForces = Kokkos::create_mirror_view(Kokkos::subview(oldForces, cellNumber, Kokkos::ALL));
  auto h_particleIDs = Kokkos::create_mirror_view(Kokkos::subview(particleIDs, cellNumber, Kokkos::ALL));
  auto h_typeIDs = Kokkos::create_mirror_view(Kokkos::subview(typeIDs, cellNumber, Kokkos::ALL));

  Kokkos::deep_copy(h_positions, Kokkos::subview(positions, cellNumber, Kokkos::ALL));
  Kokkos::deep_copy(h_velocities, Kokkos::subview(velocities, cellNumber, Kokkos::ALL));
  Kokkos::deep_copy(h_forces, Kokkos::subview(forces, cellNumber, Kokkos::ALL));
  Kokkos::deep_copy(h_oldForces, Kokkos::subview(oldForces, cellNumber, Kokkos::ALL));
  Kokkos::deep_copy(h_particleIDs, Kokkos::subview(particleIDs, cellNumber, Kokkos::ALL));
  Kokkos::deep_copy(h_typeIDs, Kokkos::subview(typeIDs, cellNumber, Kokkos::ALL));

  particles.reserve(cellSizes.view_host()(cellNumber));
  for (int iParticle = 0; iParticle < cellSizes.view_host()(cellNumber); ++iParticle) {
    particles.emplace_back(
        h_particleIDs(iParticle),
        h_typeIDs(iParticle),
        h_positions(iParticle),
        h_forces(iParticle),
        h_velocities(iParticle),
        h_oldForces(iParticle)
    );
  }
  return particles;
}

std::vector<Particle> Simulation::getParticles() const {
  std::vector<Particle> particles;
  for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
    for (const auto &particle : getParticles(cellNumber)) {
      particles.push_back(particle);
    }
  }
  return particles;
}

void Simulation::removeParticle(int cellNumber, int index) {
  int &size = cellSizes.view_host()(cellNumber);
  --size;
  cellSizes.modify_host();
  Kokkos::deep_copy(Kokkos::subview(positions, cellNumber, index), Kokkos::subview(positions, cellNumber, size));
  Kokkos::deep_copy(Kokkos::subview(velocities, cellNumber, index), Kokkos::subview(velocities, cellNumber, size));
  Kokkos::deep_copy(Kokkos::subview(forces, cellNumber, index), Kokkos::subview(forces, cellNumber, size));
  Kokkos::deep_copy(Kokkos::subview(oldForces, cellNumber, index), Kokkos::subview(oldForces, cellNumber, size));
  Kokkos::deep_copy(Kokkos::subview(particleIDs, cellNumber, index), Kokkos::subview(particleIDs, cellNumber, size));
  Kokkos::deep_copy(Kokkos::subview(typeIDs, cellNumber, index), Kokkos::subview(typeIDs, cellNumber, size));
}

void Simulation::calculateForces() const {
  /*
//TODO get from particlePropertiesLibrary
  const double epsilon = 1;
  const double sigma = 1;
  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  const auto calculator = [=](const Coord3D &distance) {
    const double distanceValue = distance.absoluteValue();
    if (distanceValue > config.cutoff) {
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
  };

  // Save oldForces and initialize new forces
  Kokkos::parallel_for(
      "saveOldForce",
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, numCells),
      KOKKOS_LAMBDA(int index) {
        const Cell &cell = cells(index);
        for (int i = 0; i < cell.size; ++i) {
          cell.oldForceAt(i) = cell.forceAt(i);
          cell.forceAt(i) = config.globalForce;
        }
      }
  );

  Kokkos::parallel_for(
      "calculateForces",
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, numCells),
      KOKKOS_LAMBDA(const int cellNumber) {
        const Cell &cell = cells(cellNumber);
        if (cell.isHaloCell) {
          return;
        }
        for (int neighbour = 0; neighbour < 27; ++neighbour) {
          const int neighbourCellNumber = neighbours(cellNumber, neighbour);
          if (cells(neighbourCellNumber).isHaloCell) {
            switch (boundaryCondition) {
              case none:break;
              case periodic: {
                const int periodicTargetCellNumber = periodicTargetCellNumbers(neighbourCellNumber);
                const Coord3D offset =
                    cells(periodicTargetCellNumber).bottomLeftCorner.distanceTo(cells(neighbourCellNumber).bottomLeftCorner);
                const Cell &neighbourCell = cells(periodicTargetCellNumber);
                for (int id_1 = 0; id_1 < cell.size; ++id_1) {
                  for (int id_2 = 0; id_2 < neighbourCell.size; ++id_2) {
                    if (cell.particleIDAt(id_1) != neighbourCell.particleIDAt(id_2)) {
                      cell.forceAt(id_1) +=
                          calculator(cell.positionAt(id_1).distanceTo(neighbourCell.positionAt(id_2) + offset));
                    }
                  }
                }
                break;
              }
              case reflecting: {
                const Cell &neighbourCell = cells(neighbourCellNumber);
                const Coord3D cellOffset = cell.bottomLeftCorner.distanceTo(neighbourCell.bottomLeftCorner);
                // If more than one of the cells coordinates are different, the neighbour cell is an edge or a corner
                if (std::abs(cellOffset.x) + std::abs(cellOffset.y) + std::abs(cellOffset.z) > config.cutoff) {
                  continue;
                }
                for (int id = 0; id < cell.size; ++id) {
                  const Coord3D position = cell.positionAt(id);
                  Coord3D ghostPosition = position + cellOffset;
                  if (cellOffset.x != 0) {
                    ghostPosition.x = neighbourCell.bottomLeftCorner.x
                        + (config.cutoff - (ghostPosition.x - neighbourCell.bottomLeftCorner.x));
                  } else if (cellOffset.y != 0) {
                    ghostPosition.y = neighbourCell.bottomLeftCorner.y
                        + (config.cutoff - (ghostPosition.y - neighbourCell.bottomLeftCorner.y));
                  } else if (cellOffset.z != 0) {
                    ghostPosition.z = neighbourCell.bottomLeftCorner.z
                        + (config.cutoff - (ghostPosition.z - neighbourCell.bottomLeftCorner.z));
                  }
                  cell.forceAt(id) += calculator(position.distanceTo(ghostPosition));
                }
                break;
              }
            }
          } else {
            const Cell &neighbourCell = cells(neighbourCellNumber);
            for (int id_1 = 0; id_1 < cell.size; ++id_1) {
              for (int id_2 = 0; id_2 < neighbourCell.size; ++id_2) {
                if (cell.particleIDAt(id_1) != neighbourCell.particleIDAt(id_2)) {
                  cell.forceAt(id_1) +=
                      calculator(cell.positionAt(id_1).distanceTo(neighbourCell.positionAt(id_2)));
                }
              }
            }
          }
        }
      }
  );
   */
}

void Simulation::calculateForcesNewton3() const {
  //TODO get from particlePropertiesLibrary
  constexpr double epsilon = 1;
  constexpr double sigma = 1;
  constexpr double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  constexpr double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  constexpr double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  const auto cellSizesCopy = cellSizes;
  const auto positionsCopy = positions;
  const auto forcesCopy = forces;
  const auto numCellsXCopy = numCellsX;
  const auto numCellsYCopy = numCellsY;
  const auto numCellsZCopy = numCellsZ;
  const auto cutoffCopy = config.cutoff;
  const auto c08PairsCopy = c08Pairs;
  const auto periodicTargetCellNumbersCopy = periodicTargetCellNumbers;
  const auto isHaloCopy = isHalo;
  const auto bottomLeftCornersCopy = bottomLeftCorners;
  const auto oldForcesCopy = oldForces;
  const auto globalForceCopy = config.globalForce;

  // Save oldForces and initialize new forces
  Kokkos::parallel_for(
      "saveAndCopyForces  Iteration: " + std::to_string(iteration),
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, numCells),
      KOKKOS_LAMBDA(int cellNumber) {
        for (int i = 0; i < cellSizesCopy.view_device()(cellNumber); ++i) {
          oldForcesCopy(cellNumber,i) = forcesCopy(cellNumber,i);
          forcesCopy(cellNumber,i) = globalForceCopy;
        }
      }
  );
  Kokkos::fence();

  for (int color = 0; color < 8; ++color) {
    const auto colorCells = c08baseCells[color];
    Kokkos::parallel_for(
        "calculateForcesForColor: " + std::to_string(color) + "  Iteration: " + std::to_string(iteration),
        Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, colorCells.size()),
        KOKKOS_LAMBDA(const int index) {
          const int baseCellNumber = colorCells(index);
          int relativeX, relativeY, relativeZ;
          getRelativeCellCoordinatesDevice(baseCellNumber,
                                           numCellsXCopy,
                                           numCellsYCopy,
                                           relativeX,
                                           relativeY,
                                           relativeZ);
          if (relativeX == numCellsXCopy - 1 ||
              relativeY == numCellsYCopy - 1 ||
              relativeZ == numCellsZCopy - 1) {
            return;
          }
          for (int id_1 = 0; id_1 < cellSizesCopy.view_device()(baseCellNumber); ++id_1) {
            for (int id_2 = id_1 + 1; id_2 < cellSizesCopy.view_device()(baseCellNumber); ++id_2) {

              const Coord3D actingForce = calculator(
                  positionsCopy(baseCellNumber,id_1).distanceTo(positionsCopy(baseCellNumber,id_2)),
                  cutoffCopy
              );
              forcesCopy(baseCellNumber,id_1) += actingForce;
              forcesCopy(baseCellNumber,id_2) += actingForce * (-1);
            }
          }
          for (int pairNumber = 0; pairNumber < 13; ++pairNumber) {
            const int cellOneNumber = c08PairsCopy(baseCellNumber, pairNumber, 0);
            const int cellTwoNumber = c08PairsCopy(baseCellNumber, pairNumber, 1);
            if (!isHaloCopy(cellOneNumber) && !isHaloCopy(cellTwoNumber)) {
              for (int id_1 = 0; id_1 < cellSizesCopy.view_device()(cellOneNumber); ++id_1) {
                for (int id_2 = 0; id_2 < cellSizesCopy.view_device()(cellTwoNumber); ++id_2) {
                  const Coord3D actingForce = calculator(
                      positionsCopy(cellOneNumber,id_1).distanceTo(positionsCopy(cellTwoNumber,id_2)),
                      cutoffCopy
                  );
                  forcesCopy(cellOneNumber,id_1) += actingForce;
                  forcesCopy(cellTwoNumber,id_2) += actingForce * (-1);
                }
              }
            } else {
              if (isHaloCopy(cellOneNumber) && isHaloCopy(cellTwoNumber)) {
                continue;
              }
              const int normalCellNumber = isHaloCopy(cellOneNumber) ? cellTwoNumber : cellOneNumber;
              const int haloCellNumber = isHaloCopy(cellOneNumber) ? cellOneNumber : cellTwoNumber;
              switch (boundaryCondition) {
                case none:break;
                case periodic: {
                  const int periodicTargetCellNumber = periodicTargetCellNumbersCopy(haloCellNumber);
                  const Coord3D offset =
                      bottomLeftCornersCopy(periodicTargetCellNumber).distanceTo(bottomLeftCornersCopy(haloCellNumber));
                  for (int id_1 = 0; id_1 < cellSizesCopy.view_device()(normalCellNumber); ++id_1) {
                    for (int id_2 = 0; id_2 < cellSizesCopy.view_device()(periodicTargetCellNumber); ++id_2) {
                      forcesCopy(normalCellNumber,id_1) += calculator(
                          positionsCopy(normalCellNumber,id_1).distanceTo(
                              positionsCopy(periodicTargetCellNumber,id_2)
                          ) + offset,
                          cutoffCopy
                      );
                    }
                  }
                  break;
                }
                case reflecting: {
                  const Coord3D offset =
                      bottomLeftCornersCopy(normalCellNumber).distanceTo(bottomLeftCornersCopy(haloCellNumber));
                  if (std::abs(offset.x) + std::abs(offset.y) + std::abs(offset.z) > cutoffCopy) {
                    continue;
                  }
                  for (int id = 0; id < cellSizesCopy.view_device()(normalCellNumber); ++id) {
                    const Coord3D position = positionsCopy(normalCellNumber,id);
                    Coord3D ghostPosition = position + offset;
                    const auto haloCellBottomLeftCorner = bottomLeftCornersCopy(haloCellNumber);
                    if (offset.x != 0) {
                      ghostPosition.x = haloCellBottomLeftCorner.x
                          + (cutoffCopy - (ghostPosition.x - haloCellBottomLeftCorner.x));
                    } else if (offset.y != 0) {
                      ghostPosition.y = haloCellBottomLeftCorner.y
                          + (cutoffCopy - (ghostPosition.y - haloCellBottomLeftCorner.y));
                    } else if (offset.z != 0) {
                      ghostPosition.z = haloCellBottomLeftCorner.z
                          + (cutoffCopy - (ghostPosition.z - haloCellBottomLeftCorner.z));
                    }
                    forcesCopy(normalCellNumber,id) += calculator(position.distanceTo(ghostPosition), cutoffCopy);
                  }
                  break;
                }
              }
            }
          }
        }
    );
    Kokkos::fence();
  }
}

void Simulation::calculateVelocitiesAndPositions() const {

  const auto cellSizesCopy = cellSizes;
  const auto forcesCopy = forces;
  const auto oldForcesCopy = oldForces;
  const auto typeIDsCopy = typeIDs;
  const auto deltaTCopy = config.deltaT;
  const auto particlePropertiesCopy = particleProperties;
  const auto positionsCopy = positions;
  const auto velocitiesCopy = velocities;
  const auto boxMinCopy = boxMin;
  const auto cutoffCopy = config.cutoff;
  const auto numCellsXCopy = numCellsX;
  const auto numCellsYCopy = numCellsY;
  const auto hasMovedCopy = hasMoved;

  Kokkos::parallel_for(
      "calculateVelocitiesAndPosition  Iteration: " + std::to_string(iteration),
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, numCells),
      KOKKOS_LAMBDA(int cellNumber) {
        for (int i = 0; i < cellSizesCopy.view_device()(cellNumber); ++i) {
          auto &velocity = velocitiesCopy(cellNumber,i);
          auto &position = positionsCopy(cellNumber,i);
          const auto force = forcesCopy(cellNumber,i);
          const auto oldForce = oldForcesCopy(cellNumber,i);
          const auto
              mass = particlePropertiesCopy.value_at(particlePropertiesCopy.find(typeIDsCopy(cellNumber,i))).mass;

          velocity += (force + oldForce) * (deltaTCopy / (2 * mass));
          position += velocity * deltaTCopy + force * ((deltaTCopy * deltaTCopy) / (2 * mass));

          const int correctCellNumber = getCorrectCellNumberDevice(
              position,
              boxMinCopy,
              cutoffCopy,
              numCellsXCopy,
              numCellsYCopy
          );
          if (cellNumber != correctCellNumber) {
            hasMovedCopy.view_device()(cellNumber) = true;
          }
        }
      }
  );
  Kokkos::fence();
}

void Simulation::moveParticles() {

  const auto numCellsXCopy = numCellsX;
  const auto numCellsYCopy = numCellsY;
  hasMoved.modify_device();
  hasMoved.sync_host();

  for (int x = 1; x < numCellsX - 1; ++x) {
    for (int y = 1; y < numCellsY - 1; ++y) {
      for (int z = 1; z < numCellsZ - 1; ++z) {
        const int cellNumber = getCellNumberFromRelativeCellCoordinates(x, y, z);
        if (!hasMoved.view_host()(cellNumber)) {
          continue;
        }
        hasMoved.view_host()(cellNumber) = false;
        hasMoved.modify_host();
        spdlog::info("before");
        const auto particles = getParticles(cellNumber);
        spdlog::info("after");
        for (int particleIndex = particles.size() - 1; particleIndex >= 0; --particleIndex) {
          Particle particle = particles[particleIndex];
          const int correctCellNumber = getCorrectCellNumber(particle);
          if (cellNumber == correctCellNumber) {
            continue;
          }
          removeParticle(cellNumber, particleIndex);
          if (correctCellNumber < 0 || numCells <= correctCellNumber) {
            std::cout
                << "A particle escaped the simulation. Most likely, this happened because the particle was moving fast "
                << "enough to pass the halo cell layer in one time step."
                << std::endl;
            exit(1);
          }
          if (h_isHalo(correctCellNumber)) {
            switch (boundaryCondition) {
              case none:
              case reflecting:break;
              case periodic: {
                int relativeX, relativeY, relativeZ;
                getRelativeCellCoordinatesDevice(correctCellNumber,
                                                 numCellsXCopy,
                                                 numCellsYCopy,
                                                 relativeX,
                                                 relativeY,
                                                 relativeZ);
                const int correctX = relativeX;
                const int correctY = relativeY;
                const int correctZ = relativeZ;
                particle.position += Coord3D(
                    (correctX == 0 ? 1 : correctX == numCellsX - 1 ? -1 : 0) * config.cutoff * (numCellsX - 2),
                    (correctY == 0 ? 1 : correctY == numCellsY - 1 ? -1 : 0) * config.cutoff * (numCellsY - 2),
                    (correctZ == 0 ? 1 : correctZ == numCellsZ - 1 ? -1 : 0) * config.cutoff * (numCellsZ - 2)
                );
                addParticle(particle);
              }
                break;
            }
          } else {
            addParticle(particle);
          }
        }
      }
    }
  }
  hasMoved.sync_device();
  cellSizes.sync_device();
}

int Simulation::getCellNumberFromRelativeCellCoordinates(const int x, const int y, const int z) const {
  return z * numCellsX * numCellsY + y * numCellsX + x;
}

std::array<int, 3> Simulation::getRelativeCellCoordinates(int cellNumber) const {
  int z = cellNumber / (numCellsX * numCellsY);
  cellNumber -= z * (numCellsX * numCellsY);
  int y = cellNumber / numCellsX;
  cellNumber -= y * numCellsX;
  return {cellNumber, y, z};
}

std::vector<int> Simulation::getNeighbourCellNumbers(const int cellNumber) const {
  std::vector<int> neighbourNumbers;
  const auto coords = getRelativeCellCoordinates(cellNumber);
  for (int z = coords[2] - 1; z <= coords[2] + 1; ++z) {
    for (int y = coords[1] - 1; y <= coords[1] + 1; ++y) {
      for (int x = coords[0] - 1; x <= coords[0] + 1; ++x) {
        if (0 <= x && x < numCellsX && 0 <= y && y < numCellsY && 0 <= z && z < numCellsZ) {
          neighbourNumbers.push_back(getCellNumberFromRelativeCellCoordinates(x, y, z));
        }
      }
    }
  }
  return neighbourNumbers;
}

int Simulation::getCorrectCellNumber(const Particle &particle) const {
  const Coord3D cellPosition = (particle.position - boxMin) / config.cutoff;
  return getCellNumberFromRelativeCellCoordinates(static_cast<int>(cellPosition.x),
                                                  static_cast<int>(cellPosition.y),
                                                  static_cast<int>(cellPosition.z));
}

int Simulation::getCellColor(const int cellNumber) const {
  const auto coords = getRelativeCellCoordinates(cellNumber);
  return (coords[0] % 2 == 0 ? 0 : 1) + (coords[1] % 2 == 0 ? 0 : 2) + (coords[2] % 2 == 0 ? 0 : 4);
}

void Simulation::writeVTKFile(const std::string &fileBaseName) const {
  if (!config.vtk) {
    return;
  }
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(config.iterations).length();
  std::vector<Particle> particles = getParticles();
  std::sort(particles.begin(), particles.end(), [](Particle &p1, Particle &p2) {
    return p1.particleID < p2.particleID;
  });
  strstr << fileBaseName << "_" << std::setfill('0') << std::setw(maxNumDigits) << iteration << ".vtk";
  std::ofstream vtkFile;
  vtkFile.open(strstr.str());

  if (not vtkFile.is_open()) {
    throw std::runtime_error("Simulation::writeVTKFile(): Failed to open file \"" + strstr.str() + "\"");
  }

  vtkFile << "# vtk DataFile Version 2.0" << "\n";
  vtkFile << "Timestep" << "\n";
  vtkFile << "ASCII" << "\n";

  // print positions
  vtkFile << "DATASET STRUCTURED_GRID" << "\n";
  vtkFile << "DIMENSIONS 1 1 1" << "\n";
  vtkFile << "POINTS " << particles.size() << " double" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].position;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }
  vtkFile << "\n";

  vtkFile << "POINT_DATA " << particles.size() << "\n";
  // print velocities
  vtkFile << "VECTORS velocities double" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].velocity;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }
  vtkFile << "\n";

  // print Forces
  vtkFile << "VECTORS forces double" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].force;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }
  vtkFile << "\n";

  // print TypeIDs
  vtkFile << "SCALARS typeIds int" << "\n";
  vtkFile << "LOOKUP_TABLE default" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << particles[i].typeID << "\n";
  }
  vtkFile << "\n";

  // print ParticleIDs
  vtkFile << "SCALARS particleIds int" << "\n";
  vtkFile << "LOOKUP_TABLE default" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << particles[i].particleID << "\n";
  }
  vtkFile << std::endl;
  vtkFile.close();
}

void Simulation::initializeSimulation() {
  std::cout << "Using the following simulation configuration:\n\n" << config << "\n" << std::endl;
  spdlog::info("Initializing particles...");
  Kokkos::Timer timer;

  std::vector<Particle> particles;

  /*
   * All particles are extracted from the particle groups and saved inside an std::vector. At the same time the
   * Kokkos::UnorderedMap particleProperties is constructed and filled with all new particleTypes. The size of the map
   * is set to the amount of particleGroups as this is the maximum possible amount of different particleTypes for the
   * simulation. In reality, this number is a lot smaller because many particleGroups will have the same particleType.
   */
  {
    particleProperties = Kokkos::UnorderedMap<int, ParticleProperties>(config.particleGroups.size());
    const auto particlePropertiesCopy = particleProperties;
    for (const auto &particleGroup : config.particleGroups) {
      const int typeID = particleGroup->typeID;
      const ParticleProperties pp(particleGroup->particleMass);
      Kokkos::parallel_for("add particle properties", 1, KOKKOS_LAMBDA(int i) {
        if (!particlePropertiesCopy.exists(typeID)) {
          particlePropertiesCopy.insert(typeID, pp);
        }
      });
      Kokkos::fence();
      const auto newParticles = particleGroup->getParticles(particles.size());
      for (const auto &particle : newParticles) {
        particles.push_back(particle);
      }
    }
  }

  /*
   * If values for the size of the simulation space are given, they are used. Otherwise size values are created by
   * starting with one cell that is positioned at the mean position of all particles. Afterwards, more cells are added
   * in all 3 spacial directions until the cells together contain all particles. The cells are added in all 3 directions
   * separately, so the simulation space is not necessarily a cube, but always a cuboid.
   */
  {
    if (config.box) {
      boxMin = config.box.value().first;
      boxMax = config.box.value().second;
    } else {
      double lowestX = particles[0].position.x;
      double lowestY = particles[0].position.y;
      double lowestZ = particles[0].position.z;
      double highestX = particles[0].position.x;
      double highestY = particles[0].position.y;
      double highestZ = particles[0].position.z;
      Coord3D midPoint = Coord3D();
      for (auto &particle : particles) {
        lowestX = std::min(lowestX, particle.position.x);
        lowestY = std::min(lowestY, particle.position.y);
        lowestZ = std::min(lowestZ, particle.position.z);
        highestX = std::max(highestX, particle.position.x);
        highestY = std::max(highestY, particle.position.y);
        highestZ = std::max(highestZ, particle.position.z);
        midPoint += particle.position;
      }
      midPoint /= particles.size();
      boxMin = midPoint - Coord3D(config.cutoff, config.cutoff, config.cutoff);
      boxMax = midPoint;
      while (lowestX < boxMin.x) {
        boxMin = boxMin - Coord3D(config.cutoff, 0, 0);
      }
      while (boxMax.x + config.cutoff < highestX) {
        boxMax = boxMax + Coord3D(config.cutoff, 0, 0);
      }
      while (lowestY < boxMin.y) {
        boxMin = boxMin - Coord3D(0, config.cutoff, 0);
      }
      while (boxMax.y + config.cutoff < highestY) {
        boxMax = boxMax + Coord3D(0, config.cutoff, 0);
      }
      while (lowestZ < boxMin.z) {
        boxMin = boxMin - Coord3D(0, 0, config.cutoff);
      }
      while (boxMax.z + config.cutoff < highestZ) {
        boxMax = boxMax + Coord3D(0, 0, config.cutoff);
      }
    }

    // A one cell wide layer of empty cells is added around the simulation space to act as halo cells in the simulation.
    boxMin = boxMin - Coord3D(config.cutoff, config.cutoff, config.cutoff);
    boxMax = boxMax + Coord3D(config.cutoff, config.cutoff, config.cutoff);

    // The total size of the simulation, and with it the number of cells is calculated.
    const Coord3D boxSize = boxMax - boxMin + Coord3D(config.cutoff, config.cutoff, config.cutoff);
    numCellsX = static_cast<int>((boxSize.x + 0.5 * config.cutoff) / config.cutoff);
    numCellsY = static_cast<int>((boxSize.y + 0.5 * config.cutoff) / config.cutoff);
    numCellsZ = static_cast<int>((boxSize.z + 0.5 * config.cutoff) / config.cutoff);
    numCells = numCellsX * numCellsY * numCellsZ;
  }

  /*
   * The cells and periodicTargetCellNumbers views are initialized to the correct size. The periodicTargetCellNumbers
   * view will only be used if periodic boundary conditions are used in the simulation, but is still always created. It
   * has a size of numCells, although it only needs to store information for the halo cells. Both actions do not create
   * too much computational overhead, while simplifying the code a lot.
   * The cells view has to be created with the Kokkos::WithoutInitializing flag because it will contain other views.
   */
  {
    cellSizes = Kokkos::DualView<int *>("cellSizes", numCells);
    hasMoved = Kokkos::DualView<bool *>("hasMoved", numCells);
    const auto cellSizesCopy = cellSizes;
    const auto hasMovedCopy = hasMoved;
    Kokkos::parallel_for("initSizesAndCapacities", numCells, KOKKOS_LAMBDA(int i) {
      cellSizesCopy.view_device()(i) = 0;
      hasMovedCopy.view_device()(i) = false;
    });
    for (int i = 0; i < numCells; ++i) {
      cellSizes.view_host()(i) = 0;
      hasMoved.view_host()(i) = false;
    }
    Kokkos::fence();

    positions = Kokkos::View<Coord3D **>("positions", numCells, capacities);
    forces = Kokkos::View<Coord3D **>("forces", numCells, capacities);
    oldForces = Kokkos::View<Coord3D **>("oldForces", numCells, capacities);
    velocities = Kokkos::View<Coord3D **>("velocities", numCells, capacities);
    typeIDs = Kokkos::View<int **>("typeIDs", numCells, capacities);
    particleIDs = Kokkos::View<int **>("particleIDs", numCells, capacities);

    //cells = CellsViewType(Kokkos::view_alloc(std::string("Cells"), Kokkos::WithoutInitializing), numCells);
    Kokkos::fence();
    periodicTargetCellNumbers = Kokkos::View<int *>("periodicTargetCellNumbers", numCells);
    isHalo = Kokkos::View<bool *>("haloCells", numCells);
    bottomLeftCorners = Kokkos::View<Coord3D *>("bottomLeftCorners", numCells);
    auto h_periodicTargetCellNumbers = Kokkos::create_mirror_view(periodicTargetCellNumbers);
    h_isHalo = Kokkos::create_mirror_view(isHalo);
    auto h_bottomLeftCorners = Kokkos::create_mirror_view(bottomLeftCorners);
    Kokkos::deep_copy(h_periodicTargetCellNumbers, periodicTargetCellNumbers);
    Kokkos::deep_copy(h_isHalo, isHalo);
    Kokkos::deep_copy(h_bottomLeftCorners, bottomLeftCorners);

    // All necessary cells are created and saved in the cells view.
    for (int x = 0; x < numCellsX; ++x) {
      for (int y = 0; y < numCellsY; ++y) {
        for (int z = 0; z < numCellsZ; ++z) {
          //TODO maybe delete this
          const bool isHaloCell =
              x == 0 || x == numCellsX - 1 || y == 0 || y == numCellsY - 1 || z == 0 || z == numCellsZ - 1;
          const Coord3D bottomLeftCorner = boxMin + Coord3D(x, y, z) * config.cutoff;
          const int cellNumber = getCellNumberFromRelativeCellCoordinates(x, y, z);

          // The cells have to be saved via the new operator because they are views inside of views.
          //new(&cells[cellNumber]) Cell(1, isHaloCell, bottomLeftCorner);
          h_isHalo(cellNumber) = isHaloCell;
          h_bottomLeftCorners(cellNumber) = bottomLeftCorner;


          // For non-halo cells the periodicTargetCellNumber is equal to the cellNumber. For halo cells they are different.
          const int targetX = x == 0 ? numCellsX - 2 : x == numCellsX - 1 ? 1 : x;
          const int targetY = y == 0 ? numCellsY - 2 : y == numCellsY - 1 ? 1 : y;
          const int targetZ = z == 0 ? numCellsZ - 2 : z == numCellsZ - 1 ? 1 : z;
          const int periodicTargetCellNumber = getCellNumberFromRelativeCellCoordinates(targetX, targetY, targetZ);
          h_periodicTargetCellNumbers(cellNumber) = periodicTargetCellNumber;
        }
      }
    }
    Kokkos::deep_copy(periodicTargetCellNumbers, h_periodicTargetCellNumbers);
    Kokkos::deep_copy(isHalo, h_isHalo);
    Kokkos::deep_copy(bottomLeftCorners, h_bottomLeftCorners);
  }

  // For each cell, the neighbours view is filled with the cell numbers of its neighbours.
  {
    neighbours = Kokkos::View<int *[27]>("neighbours", numCells);
    const auto h_neighbours = Kokkos::create_mirror_view(neighbours);
    Kokkos::deep_copy(h_neighbours, neighbours);
    for (int i = 0; i < numCells; ++i) {
      std::vector<int> neighbourNumbers = getNeighbourCellNumbers(i);
      for (int k = 0; k < neighbourNumbers.size(); ++k) {
        h_neighbours(i, k) = neighbourNumbers[k];
      }
      for (int k = neighbourNumbers.size(); k < 27; ++k) {
        h_neighbours(i, k) = -1;
      }
    }
    Kokkos::deep_copy(neighbours, h_neighbours);
  }

  {
    // The cellNumbers are sorted into 8 std::vectors representing the 8 different colors of the c08 cell coloring.
    std::vector<std::vector<int>> c08baseCellsVec;
    c08baseCellsVec.resize(8);
    for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
      c08baseCellsVec[getCellColor(cellNumber)].push_back(cellNumber);
    }

    /*
     * For each of the 8 vectors of cellNumbers representing the different colors, the base cell numbers are saved into
     * the corresponding view.
     */
    for (int i = 0; i < c08baseCellsVec.size(); ++i) {
      const int size = c08baseCellsVec[i].size();
      c08baseCells[i] = Kokkos::View<int *>("c08baseCells " + std::to_string(i), size);
      const auto h_c08BaseCells = Kokkos::create_mirror_view(c08baseCells[i]);
      Kokkos::deep_copy(h_c08BaseCells, c08baseCells[i]);
      for (int k = 0; k < size; ++k) {
        h_c08BaseCells(k) = c08baseCellsVec[i][k];
      }
      Kokkos::deep_copy(c08baseCells[i], h_c08BaseCells);
    }
  }


  // For each c08 base cell, all cell pairs for force interactions are saved into the c08Pairs view.
  {
    c08Pairs = Kokkos::View<int *[13][2]>("c08Pairs", numCells);
    auto h_c08Pairs = Kokkos::create_mirror_view(c08Pairs);
    Kokkos::deep_copy(h_c08Pairs, c08Pairs);
    for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
      auto coords = getRelativeCellCoordinates(cellNumber);
      int index = 0;
      // Iterate over every neighbour cell with a higher cell number.
      for (int x = coords[0] - 1; x < coords[0] + 2; ++x) {
        for (int y = coords[1] - 1; y < coords[1] + 2; ++y) {
          for (int z = coords[2] - 1; z < coords[2] + 2; ++z) {
            if (cellNumber >= getCellNumberFromRelativeCellCoordinates(x, y, z)) {
              continue;
            }
            Coord3D cellOne(coords[0], coords[1], coords[2]);
            Coord3D cellTwo(x, y, z);

            /*
             * If one of the 3 coordinates of the second cell is smaller than the coordinate of the first cell,
             * increment this coordinate for both cells.
             */
            if (x < coords[0]) {
              cellOne += Coord3D(1, 0, 0);
              cellTwo += Coord3D(1, 0, 0);
            }
            if (y < coords[1]) {
              cellOne += Coord3D(0, 1, 0);
              cellTwo += Coord3D(0, 1, 0);
            }
            if (z < coords[2]) {
              cellOne += Coord3D(0, 0, 1);
              cellTwo += Coord3D(0, 0, 1);
            }
            int cellNumberOne = getCellNumberFromRelativeCellCoordinates(cellOne.x, cellOne.y, cellOne.z);
            int cellNumberTwo = getCellNumberFromRelativeCellCoordinates(cellTwo.x, cellTwo.y, cellTwo.z);
            h_c08Pairs(cellNumber, index, 0) = cellNumberOne;
            h_c08Pairs(cellNumber, index++, 1) = cellNumberTwo;
          }
        }
      }
    }
    Kokkos::deep_copy(c08Pairs, h_c08Pairs);
  }
  Kokkos::fence();
  // After all cells are initialized, the particles are added
  for (auto &particle : particles) {
    addParticle(particle);
  }
  Kokkos::fence();
  const double time = timer.seconds();
  spdlog::info("Finished initializing " + std::to_string(particles.size()) + " particles. Time: "
                   + std::to_string(time) + " seconds.");
}
