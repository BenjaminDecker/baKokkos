//
// Created by Benjamin Decker on 08.11.20.
//

#include "LinkedCellsParticleContainer.h"
#include <spdlog/spdlog.h>
#include <iomanip>
#include <fstream>

LinkedCellsParticleContainer::LinkedCellsParticleContainer(const std::vector<Particle> &particles,
                                                           const SimulationConfig &config)
    : cutoff(config.cutoff), vtkFilename(config.vtkFileName) {
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
    boxMin = midPoint + ((boxMin - midPoint) * 2);
    boxMax = midPoint + ((boxMax - midPoint) * 2);
  }

  Coord3D boxSize = boxMax - boxMin;
  numCellsX = std::max(1, static_cast<int>(boxSize.x / cutoff));
  numCellsY = std::max(1, static_cast<int>(boxSize.y / cutoff));
  numCellsZ = std::max(1, static_cast<int>(boxSize.z / cutoff));
  numCells = numCellsX * numCellsY * numCellsZ;

  sizesAndCapacities = SizesAndCapacitiesType("sizedAndCapacities", numCells);
  // Set starting capacity of all cells to 1
  for (int i = 0; i < numCells; ++i) {
    sizesAndCapacities(i, 1) = 1;
  }

  cells = ContainerViewType(Kokkos::view_alloc(std::string("Cells"), Kokkos::WithoutInitializing), numCells);
  for (int i = 0; i < numCells; ++i) {
    const std::string label = std::string("Cell ") + std::to_string(i);
    new(&cells[i]) CellViewType(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), sizesAndCapacities(i, 1));
  }

  for (auto &particle : particles) {
    addParticle(particle);
  }

  neighbours = Kokkos::View<int *[27]>("neighbours", cells.size());
  auto h_neighbours = Kokkos::create_mirror_view(neighbours);
  for (int i = 0; i < cells.size(); ++i) {
    auto neighbourNumbers = getNeighbourCellNumbers(i);
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
    c08baseCells[i] = Kokkos::View<int*>("c08baseCells " + std::to_string(i), size);
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
    if (sizesAndCapacities(cellNumber, 0) == sizesAndCapacities(cellNumber, 1)) {
      resizeCellCapacity(cellNumber, 2);
    }

    Kokkos::parallel_for("addParticle", 1, KOKKOS_LAMBDA(int i) {
      const int index = sizesAndCapacities(cellNumber, 0);
      cells(cellNumber)(index, ParticleIndices::position) = particle.position;
      cells(cellNumber)(index, ParticleIndices::velocity) = particle.velocity;
      cells(cellNumber)(index, ParticleIndices::force) = particle.force;
      cells(cellNumber)(index, ParticleIndices::oldForce) = particle.oldForce;
      cells(cellNumber)(index, ParticleIndices::particleIDAndTypeID) = Coord3D(particle.particleID, particle.typeID, 0);
    });
    ++sizesAndCapacities(cellNumber, 0);
  }
}

std::vector<Particle> LinkedCellsParticleContainer::getParticles() const {
  std::vector<Particle> particles;
  for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
    for (auto &particle : getParticles(cellNumber)) {
      particles.push_back(particle);
    }
  }
  return particles;
}

std::vector<Particle> LinkedCellsParticleContainer::getParticles(int cellNumber) const {
  std::vector<Particle> particles;
  auto h_particles = Kokkos::create_mirror_view(cells(cellNumber));
  const int numParticles = sizesAndCapacities(cellNumber, 0);
  for (int particleNumber = 0; particleNumber < numParticles; ++particleNumber) {
    Coord3D particleIDAndTypeID(h_particles(particleNumber, ParticleIndices::particleIDAndTypeID));
    particles.emplace_back(particleIDAndTypeID.x,
                           particleIDAndTypeID.y,
                           h_particles(particleNumber, ParticleIndices::position),
                           h_particles(particleNumber, ParticleIndices::force),
                           h_particles(particleNumber, ParticleIndices::velocity),
                           h_particles(particleNumber, ParticleIndices::oldForce));
  }
  return particles;
}

void LinkedCellsParticleContainer::iterateCalculatePositions(double deltaT) const {
  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  Kokkos::parallel_for("iterateCalculatePositions", numCells, KOKKOS_LAMBDA(int cellNumber) {
    const int numParticles = sizesAndCapacities(cellNumber, 0);
    for (int particleNumber = 0; particleNumber < numParticles; ++particleNumber) {
      cells(cellNumber)(particleNumber, ParticleIndices::position) +=
          cells(cellNumber)(particleNumber, ParticleIndices::velocity) * deltaT
              + cells(cellNumber)(particleNumber, ParticleIndices::force) * ((deltaT * deltaT) / (2 * mass));
    }
  });
}

void LinkedCellsParticleContainer::iterateCalculateForces() const {
  //TODO get from particlePropertiesLibrary
  const double epsilon = 1;
  const double sigma = 1;
  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  // Iterate over each cell in parallel
  Kokkos::parallel_for("iterateCalculateForces", numCells, KOKKOS_LAMBDA(int cellIndex) {
    // Get the number of particles in current cell
    const int numParticles = sizesAndCapacities(cellIndex, 0);
    // Iterate over every particle in current cell
    for (int id_1 = 0; id_1 < numParticles; ++id_1) {
      // Force accumulator
      Coord3D force = Coord3D();
      // Iterate over the amount of possible neighbour cells
      for (int neighbour = 0; neighbour < 27; ++neighbour) {
        // Get the index into the cells view of the current neighbour cell
        const int neighbourCellIndex = neighbours(cellIndex, neighbour);
        // Test if the neighbour exists
        if (neighbourCellIndex == -1) {
          continue;
        }
        // Get the number of particles in the current neighbour cell
        const int numParticlesNeighbour = sizesAndCapacities(neighbourCellIndex, 0);
        // Iterate over every particle of the neighbour cell
        for (int id_2 = 0; id_2 < numParticlesNeighbour; ++id_2) {
          // Test if the current particle is the same as the accumulator particle
          if (cellIndex == neighbourCellIndex && id_1 == id_2) {
            continue;
          }
          const Coord3D distance =
              cells(cellIndex)(id_1, ParticleIndices::position)
                  .distanceTo(cells(neighbourCellIndex)(id_2, ParticleIndices::position));
          const double distanceValue = distance.absoluteValue();
          const double distanceValuePow6 =
              distanceValue * distanceValue * distanceValue * distanceValue * distanceValue * distanceValue;
          const double distanceValuePow13 = distanceValuePow6 * distanceValuePow6 * distanceValue;

          // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
          const double forceValue =
              (twentyFourEpsilonSigmaPow6 * distanceValuePow6 - fourtyEightEpsilonSigmaPow12) / distanceValuePow13;
          force += (distance * (forceValue / distanceValue));
        }
      }
      // Save previous forces
      cells(cellIndex)(id_1, ParticleIndices::oldForce) = cells(cellIndex)(id_1, ParticleIndices::force);
      // Save new force
      cells(cellIndex)(id_1, ParticleIndices::force) = force;
    }
  });
}

void LinkedCellsParticleContainer::iterateCalculateForcesNewton3() const {
  
}

void LinkedCellsParticleContainer::iterateCalculateVelocities(double deltaT) const {

  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  Kokkos::parallel_for("iterateCalculateVelocities", numCells, KOKKOS_LAMBDA(int cellNumber) {
    const int numParticles = sizesAndCapacities(cellNumber, 0);
    for (int particleNumber = 0; particleNumber < numParticles; ++particleNumber) {
      cells(cellNumber)(particleNumber, ParticleIndices::velocity) +=
          (cells(cellNumber)(particleNumber, ParticleIndices::force)
              + cells(cellNumber)(particleNumber, ParticleIndices::oldForce))
              * (deltaT / (2 * mass));
    }
  });
}

void LinkedCellsParticleContainer::moveParticles() const {
  for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
    auto particles = getParticles(cellNumber);
    for (int particleIndex = 0; particleIndex < particles.size(); ++particleIndex) {
      const int correctCellNumber = getCorrectCellNumber(particles[particleIndex]);
      if (cellNumber == correctCellNumber) {
        continue;
      }
      moveParticle(particleIndex, cellNumber, correctCellNumber);
      particles[particleIndex] = particles[particles.size() - 1];
      particles.pop_back();
      --particleIndex;
    }
  }
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

void LinkedCellsParticleContainer::resizeCellCapacity(int cellNumber, int factor) const {
  const int size = sizesAndCapacities(cellNumber, 0);
  const int newCapacity = sizesAndCapacities(cellNumber, 1) * factor;
  sizesAndCapacities(cellNumber, 1) = newCapacity;
  const std::string label = cells[cellNumber].label();
  CellViewType newCell = CellViewType(label, newCapacity);
  Kokkos::parallel_for("read new cell " + std::to_string(cellNumber),
                       size,
                       KOKKOS_LAMBDA(int i) {
                         for (int k = 0; k < 5; ++k) {
                           newCell(i, k) = cells(cellNumber)(i, k);
                         }
                       });
  Kokkos::fence();
  cells[cellNumber].~CellViewType();
  new(&cells[cellNumber]) CellViewType(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), newCapacity);
  Kokkos::parallel_for("write new cell " + std::to_string(cellNumber),
                       size,
                       KOKKOS_LAMBDA(int i) {
                         for (int k = 0; k < 5; ++k) {
                           cells(cellNumber)(i, k) = newCell(i, k);
                         }
                       });
}
int LinkedCellsParticleContainer::getCorrectCellNumber(const Particle &particle) const {
  const Coord3D cellPosition = (particle.position - boxMin) / cutoff;
  return getCellNumber(static_cast<int>(cellPosition.x),
                       static_cast<int>(cellPosition.y),
                       static_cast<int>(cellPosition.z));
}

void LinkedCellsParticleContainer::moveParticle(int particleIndex, int fromCell, int toCell) const {
  if (sizesAndCapacities(toCell, 0) == sizesAndCapacities(toCell, 1)) {
    resizeCellCapacity(toCell, 2);
  }
  const std::string label = "moveParticle " + std::to_string(particleIndex)
      + " from cell " + std::to_string(fromCell) + " to cell " + std::to_string(toCell);
  Kokkos::parallel_for(label, 1, KOKKOS_LAMBDA(int i) {
    const int destinationCellSize = sizesAndCapacities(toCell, 0);
    for (int j = 0; j < 5; ++j) {
      cells(toCell)(destinationCellSize, j) = cells(fromCell)(particleIndex, j);
    }
    ++sizesAndCapacities(toCell, 0);
    --sizesAndCapacities(fromCell, 0);
    const int departureCellSize = sizesAndCapacities(fromCell, 0);
    for (int j = 0; j < 5; ++j) {
      cells(fromCell)(particleIndex, j) = cells(fromCell)(departureCellSize, j);
    }
  });
}

int LinkedCellsParticleContainer::getCellColor(int cellNumber) const {
  auto coords = getRelativeCellCoordinates(cellNumber);
  return (coords[0] % 2 == 0 ? 0 : 1) + (coords[1] % 2 == 0 ? 0 : 2) + (coords[2] % 2 == 0 ? 0 : 4);
}
