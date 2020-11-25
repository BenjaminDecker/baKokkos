//
// Created by Benjamin Decker on 08.11.20.
//

#include "LinkedCellsParticleContainer.h"
#include <spdlog/spdlog.h>
#include <iomanip>
#include <fstream>

LinkedCellsParticleContainer::LinkedCellsParticleContainer(const SimulationConfig &config)
    : vtk(config.vtk),
      deltaT(config.deltaT),
      globalForce(config.globalForce),
      cutoff(config.cutoff),
      iterations(config.iterations),
      iteration(0) {
  std::cout << "Using the following simulation configuration:" << std::endl << std::endl << config << std::endl
            << std::endl;
  spdlog::info("Initializing particles...");
  Kokkos::Timer timer;
  std::vector<Particle> particles;
  for (const auto &particleGroup : config.particleGroups) {
    auto newParticles = particleGroup->getParticles(particles.size());
    for (const auto &particle : newParticles) {
      particles.push_back(particle);
    }
  }
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
  boxMin = boxMin - Coord3D(config.cutoff, config.cutoff, config.cutoff);
  boxMax = boxMax + Coord3D(config.cutoff, config.cutoff, config.cutoff);

  Coord3D boxSize = boxMax - boxMin + Coord3D(config.cutoff, config.cutoff, config.cutoff);
  numCellsX = static_cast<int>((boxSize.x + 0.5 * config.cutoff) / config.cutoff);
  numCellsY = static_cast<int>((boxSize.y + 0.5 * config.cutoff) / config.cutoff);
  numCellsZ = static_cast<int>((boxSize.z + 0.5 * config.cutoff) / config.cutoff);
  numCells = numCellsX * numCellsY * numCellsZ;

  cells = CellsViewType(Kokkos::view_alloc(std::string("Cells"), Kokkos::WithoutInitializing), numCells);

  periodicTargetCellNumbers = Kokkos::View<int *>("periodicTargetCellNumbers", numCells);

  for (int x = 0; x < numCellsX; ++x) {
    for (int y = 0; y < numCellsY; ++y) {
      for (int z = 0; z < numCellsZ; ++z) {
        bool isHaloCell = x == 0 || x == numCellsX - 1 || y == 0 || y == numCellsY - 1 || z == 0 || z == numCellsZ - 1;
        Coord3D bottomLeftCorner = boxMin + Coord3D(x, y, z) * config.cutoff;
        const int cellNumber = getCellNumberFromRelativeCellCoordinates(x, y, z);
        new(&cells[cellNumber]) Cell(1, isHaloCell, bottomLeftCorner);
        const int targetX = x == 0 ? numCellsX - 2 : x == numCellsX - 1 ? 1 : x;
        const int targetY = y == 0 ? numCellsY - 2 : y == numCellsY - 1 ? 1 : y;
        const int targetZ = z == 0 ? numCellsZ - 2 : z == numCellsZ - 1 ? 1 : z;
        const int periodicTargetCellNumber = getCellNumberFromRelativeCellCoordinates(targetX, targetY, targetZ);
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i) {
          periodicTargetCellNumbers(cellNumber) = periodicTargetCellNumber;
        });
      }
    }
  }
  for (auto &particle : particles) {
    addParticle(particle);
  }

  neighbours = Kokkos::View<int *[27]>("neighbours", cells.size());
  auto h_neighbours = Kokkos::create_mirror_view(neighbours);
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

  std::vector<std::vector<int>> c08baseCellsVec;
  c08baseCellsVec.resize(8);
  for (int x = 0; x < numCellsX; ++x) {
    for (int y = 0; y < numCellsY; ++y) {
      for (int z = 0; z < numCellsZ; ++z) {
        int cellNumber = getCellNumberFromRelativeCellCoordinates(x, y, z);
        c08baseCellsVec[getCellColor(cellNumber)].push_back(cellNumber);
      }
    }
  }

  for (int i = 0; i < c08baseCells.size(); ++i) {
    int size = c08baseCellsVec[i].size();
    c08baseCells[i] = Kokkos::View<int *>("c08baseCells " + std::to_string(i), size);
    auto h_c08BaseCells = Kokkos::create_mirror_view(c08baseCells[i]);
    for (int k = 0; k < size; ++k) {
      h_c08BaseCells(k) = c08baseCellsVec[i][k];
    }
    Kokkos::deep_copy(c08baseCells[i], h_c08BaseCells);
  }

  const double time = timer.seconds();
  spdlog::info("Finished initializing " + std::to_string(particles.size()) + " particles. Time: "
                   + std::to_string(time) + " seconds.");
}

void LinkedCellsParticleContainer::addParticle(const Particle &particle) const {
  const int cellNumber = getCorrectCellNumber(particle);
  if (cellNumber < 0 || numCells <= cellNumber) {
    std::cout << "That should not happen" << std::endl;
    return;
  }
  cells(cellNumber).addParticle(particle);
}

std::vector<Particle> LinkedCellsParticleContainer::getParticles() const {
  std::vector<Particle> particles;
  for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
    for (auto &particle : cells(cellNumber).getParticles()) {
      particles.push_back(particle);
    }
  }
  return particles;
}

void LinkedCellsParticleContainer::doIteration() {
  calculatePositions();
  calculateForces();
  calculateVelocities();
  moveParticles();
  if (vtk && iteration % vtk.value().second == 0) {
    writeVTKFile(vtk.value().first);
  }
  ++iteration;
}

void LinkedCellsParticleContainer::calculatePositions() const {
  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  Kokkos::parallel_for("calculatePositions", numCells, KOKKOS_LAMBDA(int cellNumber) {
    const Cell &cell = cells(cellNumber);
    for (int i = 0; i < cell.size; ++i) {
      cell.positions(i) +=
          cell.velocities(i) * deltaT + cell.forces(i) * ((deltaT * deltaT) / (2 * mass));
    }
  });
}

void LinkedCellsParticleContainer::calculateForces() const {
  //TODO get from particlePropertiesLibrary
  const double epsilon = 1;
  const double sigma = 1;
  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  // Save oldForces
  for (int cell = 0; cell < numCells; ++cell) {
    Kokkos::deep_copy(cells(cell).oldForces, cells(cell).forces);
  }

  Kokkos::parallel_for("resetForces", numCells, KOKKOS_LAMBDA(int cellNumber) {
    auto cell = cells(cellNumber);
    for (int index = 0; index < cell.size; ++index) {
      cell.forces(index) = globalForce;
    }
  });

  // Iterate over each cell in parallel
  Kokkos::parallel_for(
      "calculateForces", numCells, KOKKOS_LAMBDA(const int cellNumber) {
        const Cell &cell = cells(cellNumber);
        if (cell.isHaloCell) {
          return;
        }
        for (int neighbour = 0; neighbour < 27; ++neighbour) {
          int neighbourCellNumber = neighbours(cellNumber, neighbour);
          Coord3D offset = Coord3D();
          if (cells(neighbourCellNumber).isHaloCell) {
            switch (condition) {
              case none:break;
              case periodic: {
                const int periodicTargetCellNumber = periodicTargetCellNumbers(neighbourCellNumber);
                offset = cells(neighbourCellNumber).bottomLeftCorner - cells(periodicTargetCellNumber).bottomLeftCorner;
                neighbourCellNumber = periodicTargetCellNumber;
                break;
              }
              case reflecting:
                // TODO
                break;
            }
          }
          const Cell &neighbourCell = cells(neighbourCellNumber);
          for (int id_1 = 0; id_1 < cell.size; ++id_1) {
            for (int id_2 = 0; id_2 < neighbourCell.size; ++id_2) {
              if (cell.particleIDs(id_1) == neighbourCell.particleIDs(id_2)) {
                continue;
              }
              Coord3D distance = cell.positions(id_1).distanceTo(neighbourCell.positions(id_2) + offset);
              const double distanceValue = distance.absoluteValue();
              const double distanceValuePow6 =
                  distanceValue * distanceValue * distanceValue * distanceValue * distanceValue * distanceValue;
              const double distanceValuePow13 = distanceValuePow6 * distanceValuePow6 * distanceValue;

              // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
              const double forceValue =
                  (twentyFourEpsilonSigmaPow6 * distanceValuePow6 - fourtyEightEpsilonSigmaPow12) / distanceValuePow13;
              cell.forces(id_1) += (distance * (forceValue / distanceValue));
            }
          }
        }
      });
}

void LinkedCellsParticleContainer::calculateForcesNewton3() const {
  // TODO
}

void LinkedCellsParticleContainer::calculateVelocities() const {
  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  Kokkos::parallel_for("iterateCalculateVelocities", numCells, KOKKOS_LAMBDA(int cellNumber) {
    const Cell &cell = cells(cellNumber);
    for (int i = 0; i < cell.size; ++i) {
      cell.velocities(i) += (cell.forces(i) + cell.oldForces(i)) * (deltaT / (2 * mass));
    }
  });
}

void LinkedCellsParticleContainer::moveParticles() const {
  for (int x = 1; x < numCellsX - 1; ++x) {
    for (int y = 1; y < numCellsY - 1; ++y) {
      for (int z = 1; z < numCellsZ - 1; ++z) {
        const int cellNumber = getCellNumberFromRelativeCellCoordinates(x, y, z);
        Cell &cell = cells(cellNumber);
        const auto particles = cell.getParticles();
        for (int particleIndex = particles.size() - 1; particleIndex >= 0; --particleIndex) {
          Particle particle = particles[particleIndex];
          const int correctCellNumber = getCorrectCellNumber(particle);
          if (cellNumber == correctCellNumber) {
            continue;
          }
          cell.removeParticle(particleIndex);
          if (correctCellNumber < 0 || numCells <= correctCellNumber) {
            std::cout << "That should not happen2" << std::endl;
            continue;
          }
          Cell &correctCell = cells(correctCellNumber);
          if (correctCell.isHaloCell) {
            switch (condition) {
              case none:
              case reflecting:break;
              case periodic: {
                const auto correctCoords = getRelativeCellCoordinates(correctCellNumber);
                const int correctX = correctCoords[0];
                const int correctY = correctCoords[1];
                const int correctZ = correctCoords[2];
                particle.position += Coord3D(
                    (correctX == 0 ? 1 : correctX == numCellsX - 1 ? -1 : 0) * cutoff * (numCellsX - 2),
                    (correctY == 0 ? 1 : correctY == numCellsY - 1 ? -1 : 0) * cutoff * (numCellsY - 2),
                    (correctZ == 0 ? 1 : correctZ == numCellsZ - 1 ? -1 : 0) * cutoff * (numCellsZ - 2)
                );
                addParticle(particle);
              }
                break;
            }
          } else {
            correctCell.addParticle(particle);
          }
        }
      }
    }
  }
}

int LinkedCellsParticleContainer::getCellNumberFromRelativeCellCoordinates(int x, int y, int z) const {
  return z * numCellsX * numCellsY + y * numCellsX + x;
}

std::array<int, 3> LinkedCellsParticleContainer::getRelativeCellCoordinates(int cellNumber) const {
  int z = cellNumber / (numCellsX * numCellsY);
  cellNumber -= z * (numCellsX * numCellsY);
  int y = cellNumber / numCellsX;
  cellNumber -= y * numCellsX;
  return {cellNumber, y, z};
}

std::vector<int> LinkedCellsParticleContainer::getNeighbourCellNumbers(int cellNumber) const {
  std::vector<int> neighbourNumbers;
  auto coords = getRelativeCellCoordinates(cellNumber);
  for (int x = coords[0] - 1; x <= coords[0] + 1; ++x) {
    for (int y = coords[1] - 1; y <= coords[1] + 1; ++y) {
      for (int z = coords[2] - 1; z <= coords[2] + 1; ++z) {
        if (0 <= x && x < numCellsX && 0 <= y && y < numCellsY && 0 <= z && z < numCellsZ) {
          neighbourNumbers.push_back(getCellNumberFromRelativeCellCoordinates(x, y, z));
        }
      }
    }
  }
  return neighbourNumbers;
}

int LinkedCellsParticleContainer::getCorrectCellNumber(const Particle &particle) const {
  const Coord3D cellPosition = (particle.position - boxMin) / cutoff;
  return getCellNumberFromRelativeCellCoordinates(static_cast<int>(cellPosition.x),
                                                  static_cast<int>(cellPosition.y),
                                                  static_cast<int>(cellPosition.z));
}

int LinkedCellsParticleContainer::getCellColor(int cellNumber) const {
  auto coords = getRelativeCellCoordinates(cellNumber);
  return (coords[0] % 2 == 0 ? 0 : 1) + (coords[1] % 2 == 0 ? 0 : 2) + (coords[2] % 2 == 0 ? 0 : 4);
}

void LinkedCellsParticleContainer::writeVTKFile(const std::string &fileBaseName) const {
  if (!vtk) {
    return;
  }
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(iterations).length();
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

  vtkFile << "# vtk DataFile Version 2.0" << std::endl;
  vtkFile << "Timestep" << std::endl;
  vtkFile << "ASCII" << std::endl;

  // print positions
  vtkFile << "DATASET STRUCTURED_GRID" << std::endl;
  vtkFile << "DIMENSIONS 1 1 1" << std::endl;
  vtkFile << "POINTS " << particles.size() << " double" << std::endl;
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].position;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  vtkFile << "POINT_DATA " << particles.size() << std::endl;
  // print velocities
  vtkFile << "VECTORS velocities double" << std::endl;
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].velocity;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  // print Forces
  vtkFile << "VECTORS forces double" << std::endl;
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].force;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  // print TypeIDs
  vtkFile << "SCALARS typeIds int" << std::endl;
  vtkFile << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << particles[i].typeID << std::endl;
  }
  vtkFile << std::endl;

  // print ParticleIDs
  vtkFile << "SCALARS particleIds int" << std::endl;
  vtkFile << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << particles[i].particleID << std::endl;
  }
  vtkFile << std::endl;
  vtkFile.close();
}
