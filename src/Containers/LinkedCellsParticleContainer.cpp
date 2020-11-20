//
// Created by Benjamin Decker on 08.11.20.
//

#include "LinkedCellsParticleContainer.h"
#include <spdlog/spdlog.h>
#include <iomanip>
#include <fstream>

LinkedCellsParticleContainer::LinkedCellsParticleContainer(const std::vector<Particle> &particles,
                                                           const SimulationConfig &config)
    : config(config), iteration(0) {
  spdlog::info("Initializing particles...");
  Kokkos::Timer timer;
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
    for (auto &particle : particles) {
      lowestX = std::min(lowestX, particle.position.x);
      lowestY = std::min(lowestY, particle.position.y);
      lowestZ = std::min(lowestZ, particle.position.z);
      highestX = std::max(highestX, particle.position.x);
      highestY = std::max(highestY, particle.position.y);
      highestZ = std::max(highestZ, particle.position.z);
    }
    boxMin = Coord3D(lowestX, lowestY, lowestZ);
    boxMax = Coord3D(highestX, highestY, highestZ);
    Coord3D midPoint = (boxMin + boxMax) / 2.0;
    boxMin = midPoint + ((boxMin - midPoint) * 2.0);
    boxMax = midPoint + ((boxMax - midPoint) * 2.0);
  }

  Coord3D boxSize = boxMax - boxMin;
  numCellsX = std::max(1, static_cast<int>(boxSize.x / config.cutoff));
  numCellsY = std::max(1, static_cast<int>(boxSize.y / config.cutoff));
  numCellsZ = std::max(1, static_cast<int>(boxSize.z / config.cutoff));
  numCells = numCellsX * numCellsY * numCellsZ;

  cells = CellsViewType(Kokkos::view_alloc(std::string("Cells"), Kokkos::WithoutInitializing), numCells);
  for (int i = 0; i < numCells; ++i) {
    new(&cells[i]) Cell(1);
  }

  for (auto &particle : particles) {
    addParticle(particle);
  }

  neighbours = Kokkos::View<int *[27]>("neighbours", cells.size());
  auto h_neighbours = Kokkos::create_mirror_view(neighbours);
  for (int i = 0; i < cells.size(); ++i) {
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
        int cellNumber = getCellNumber(x, y, z);
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
  if (0 <= cellNumber && cellNumber < numCells) {
    cells(cellNumber).addParticle(particle);
  }
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
  if (config.vtkFileName && iteration % config.vtkWriteFrequency == 0) {
    writeVTKFile(iteration, config.iterations, config.vtkFileName.value());
  }
  ++iteration;
}

void LinkedCellsParticleContainer::calculatePositions() const {
  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  Kokkos::parallel_for("calculatePositions", numCells, KOKKOS_LAMBDA(int cellNumber) {
    auto cell = cells(cellNumber);
    for (int i = 0; i < cell.size; ++i) {
      cell.positions(i) +=
          cell.velocities(i) * config.deltaT + cell.forces(i) * ((config.deltaT * config.deltaT) / (2 * mass));
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
      cell.forces(index) = Coord3D();
    }
  });

  // Iterate over each cell in parallel
  Kokkos::parallel_for("calculateForces", numCells, KOKKOS_LAMBDA(int cellIndex) {
    auto cell = cells(cellIndex);
    // Iterate over every particle of the cell
    for (int id_1 = 0; id_1 < cell.size; ++id_1) {
      // Iterate over the neighbours of the cell
      for (int neighbour = 0; neighbour < 27; ++neighbour) {
        // Get the index into the cells view of the neighbour cell
        const int neighbourCellIndex = neighbours(cellIndex, neighbour);
        // Test if the neighbour exists
        if (neighbourCellIndex == -1) {
          continue;
        }
        auto neighbourCell = cells(neighbourCellIndex);
        // Iterate over every particle of the neighbour cell
        for (int id_2 = 0; id_2 < neighbourCell.size; ++id_2) {
          // Skip if the two particles are the same
          if (cell.particleIDs(id_1) == neighbourCell.particleIDs(id_2)) {
            continue;
          }
          const Coord3D distance = cell.positions(id_1).distanceTo(neighbourCell.positions(id_2));
          const double distanceValue = distance.absoluteValue();
          const double distanceValuePow6 =
              distanceValue * distanceValue * distanceValue * distanceValue * distanceValue * distanceValue;
          const double distanceValuePow13 = distanceValuePow6 * distanceValuePow6 * distanceValue;

          // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
          const double forceValue =
              (twentyFourEpsilonSigmaPow6 * distanceValuePow6 - fourtyEightEpsilonSigmaPow12) / distanceValuePow13;
          if (config.globalForce) {
            cell.forces(id_1) += (distance * (forceValue / distanceValue)) + config.globalForce.value();
          } else {
            cell.forces(id_1) += (distance * (forceValue / distanceValue));
          }
        }
      }
    }
  });
}

void LinkedCellsParticleContainer::calculateForcesNewton3() const {
//  // Save old forces
//  Kokkos::parallel_for("save old forces", numCells, KOKKOS_LAMBDA(int cellIndex) {
//    // Get the number of particles in current cell
//    const int numParticles = sizesAndCapacities(cellIndex, 0);
//    // Iterate over every particle in current cell
//    for (int id_1 = 0; id_1 < numParticles; ++id_1) {
//      // Save previous forces
//      cells(cellIndex)(id_1, ParticleIndices::oldForce) = cells(cellIndex)(id_1, ParticleIndices::force);
//    }
//  });
//  // TODO
}

void LinkedCellsParticleContainer::calculateVelocities() const {
  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  Kokkos::parallel_for("iterateCalculateVelocities", numCells, KOKKOS_LAMBDA(int cellNumber) {
    auto cell = cells(cellNumber);
    for (int i = 0; i < cell.size; ++i) {
      cell.velocities(i) += (cell.forces(i) + cell.oldForces(i)) * (config.deltaT / (2 * mass));
    }
  });
}

void LinkedCellsParticleContainer::moveParticles() const {
  for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
    auto particles = cells(cellNumber).getParticles();
    for (int particleIndex = particles.size() - 1; particleIndex >= 0; --particleIndex) {
      Particle particle = particles[particleIndex];
      int correctCellNumber = getCorrectCellNumber(particle);
      if (cellNumber != correctCellNumber) {
        cells(cellNumber).removeParticle(particleIndex);
        if (0 <= correctCellNumber && correctCellNumber < numCells) {
          cells(correctCellNumber).addParticle(particle);
        }
      }
    }
  }
}

int LinkedCellsParticleContainer::getCellNumber(int x, int y, int z) const {
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
          neighbourNumbers.push_back(getCellNumber(x, y, z));
        }
      }
    }
  }
  return neighbourNumbers;
}

int LinkedCellsParticleContainer::getCorrectCellNumber(const Particle &particle) const {
  const Coord3D cellPosition = (particle.position - boxMin) / config.cutoff;
  return getCellNumber(static_cast<int>(cellPosition.x),
                       static_cast<int>(cellPosition.y),
                       static_cast<int>(cellPosition.z));
}

int LinkedCellsParticleContainer::getCellColor(int cellNumber) const {
  auto coords = getRelativeCellCoordinates(cellNumber);
  return (coords[0] % 2 == 0 ? 0 : 1) + (coords[1] % 2 == 0 ? 0 : 2) + (coords[2] % 2 == 0 ? 0 : 4);
}

void LinkedCellsParticleContainer::writeVTKFile(int iteration, int maxIterations, const std::string &fileName) const {
  const std::string fileBaseName("baKokkos");
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(maxIterations).length();
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
