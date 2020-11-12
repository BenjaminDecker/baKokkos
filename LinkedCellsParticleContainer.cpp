//
// Created by Benjamin Decker on 08.11.20.
//

#include "LinkedCellsParticleContainer.h"
#include <spdlog/spdlog.h>
#include <iomanip>
#include <fstream>

LinkedCellsParticleContainer::LinkedCellsParticleContainer(std::vector<Particle> &particles, SimulationConfig &config)
    : cutoff(config.cutoff), vtkFilename(config.vtkFileName) {
  spdlog::info("Initializing particles...");
  Kokkos::Timer timer;
  Coord3D boxMin, boxMax;
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
  numCellsX = static_cast<int>(boxSize.x / cutoff);
  numCellsY = static_cast<int>(boxSize.y / cutoff);
  numCellsZ = static_cast<int>(boxSize.z / cutoff);
  int numCells = numCellsX * numCellsY * numCellsZ;
  cells.resize(numCells);

  for (auto &particle : particles) {
    Coord3D cellPosition = (particle.position - boxMin) / cutoff;
    int cellNumber = getIndexOf(static_cast<int>(cellPosition.x),
                                static_cast<int>(cellPosition.y),
                                static_cast<int>(cellPosition.z));
    if (0 <= cellNumber && cellNumber < numCells) {
      cells[cellNumber].addParticle(particle);
    }
  }
  const double time = timer.seconds();
  spdlog::info("Finished initializing " + std::to_string(particles.size()) + " particles. Time: "
                   + std::to_string(time) + " seconds.");
}

void LinkedCellsParticleContainer::iterateCalculatePositions(double deltaT) {
  //TODO get from particlePropertiesLibrary
  constexpr double mass = 1;
  for (auto &cell : cells) {
    Kokkos::parallel_for(cell.size, KOKKOS_LAMBDA(int i) {
      cell.positions(i) += cell.velocities(i) * deltaT + cell.forces(i) * ((deltaT * deltaT) / (2 * mass));
    });
  }
}

void LinkedCellsParticleContainer::iterateCalculateForces() {
  //TODO get from particlePropertiesLibrary
  const double epsilon = 1;
  const double sigma = 1;
  const double mass = 1;
  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = Kokkos::TeamPolicy<>::member_type;

  for (int cellNumber = 0; cellNumber < cells.size(); ++cellNumber) {
    LinkedCell thisCell = cells[cellNumber];
    std::vector<LinkedCell> neighbours = std::vector<LinkedCell>();
    neighbours.reserve(27);
    neighbours.push_back(thisCell);
    for (int neighbourNumber : getNeighbourCellNumbers(cellNumber)) {
      neighbours.push_back(cells[neighbourNumber]);
    }
    Kokkos::parallel_for("iterateCalculateForces" + std::to_string(cellNumber),
                         thisCell.size,
                         KOKKOS_LAMBDA(const int id_1) {
                           Coord3D force = Coord3D();
                           for (int cell = 0; cell < neighbours.size(); ++cell) {
                             for (int id_2 = 0; id_2 < neighbours[cell].size; ++id_2) {
                               if (cell == 0 && id_1 == id_2) {
                                 return;
                               }
                               const Coord3D
                                   distance = cells[0].positions(id_1).distanceTo(cells[cell].positions(id_2));
                               const double distanceValue = distance.absoluteValue();
                               const double distanceValuePow6 =
                                   distanceValue * distanceValue * distanceValue * distanceValue
                                       * distanceValue * distanceValue;
                               const double distanceValuePow13 =
                                   distanceValuePow6 * distanceValuePow6 * distanceValue;

                               // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
                               const double forceValue =
                                   (twentyFourEpsilonSigmaPow6 * distanceValuePow6
                                       - fourtyEightEpsilonSigmaPow12) / distanceValuePow13;

                               force += (distance * (forceValue / distanceValue));
                             }
                           }

                           // TODO calculate forces from neighbour cells
                           cells[0].oldForces(id_1) = cells[0].forces(id_1);
                           cells[0].forces(id_1) = force;
                         });
  }
}

void LinkedCellsParticleContainer::iterateCalculateVelocities(double deltaT) {
  //TODO get from particlePropertiesLibrary
  constexpr int mass = 1;
  for (auto &cell : cells) {
    Kokkos::parallel_for(cell.size, KOKKOS_LAMBDA(int i) {
      cell.velocities(i) += (cell.forces(i) + cell.oldForces(i)) * (deltaT / (2 * mass));
    });
  }
}

void LinkedCellsParticleContainer::writeVTKFile(int iteration, int maxIterations, const std::string &fileName) const {
  std::string fileBaseName("baKokkos");
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(maxIterations).length();
  std::vector<Particle> particles;
  for (auto &cell : cells) {
    for (int i = 0; i < cell.size; ++i) {
      particles.push_back(cell.getParticle(i));
    }
  }
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

  // print TypeIDs
  vtkFile << "SCALARS particleIds int" << std::endl;
  vtkFile << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << i << std::endl;
  }
  vtkFile << std::endl;
  vtkFile.close();

}

int LinkedCellsParticleContainer::getIndexOf(int x, int y, int z) const {
  return z * numCellsX * numCellsY + y * numCellsX + x;
}

std::array<int, 3> LinkedCellsParticleContainer::getCoordinates(int cellNumber) const {
  int z = cellNumber / (numCellsX * numCellsY);
  cellNumber -= z * (numCellsX * numCellsY);
  int y = cellNumber / numCellsX;
  cellNumber -= y * numCellsX;
  return {cellNumber, y, z};
}

std::vector<int> LinkedCellsParticleContainer::getNeighbourCellNumbers(int cellNumber) {
  std::vector<int> neighbours;
  auto coords = getCoordinates(cellNumber);
  for (int x = coords[0] - 1; x < coords[0] + 1; ++x) {
    for (int y = coords[1] - 1; y < coords[1] + 1; ++y) {
      for (int z = coords[2] - 1; z < coords[2] + 1; ++z) {
        if (0 <= x && x < numCellsX && 0 <= y && y < numCellsY && 0 <= z && z < numCellsZ
            && (x != coords[0] || y != coords[1] || z != coords[2])) {
          neighbours.push_back(getIndexOf(x, y, z));
        }
      }
    }
  }
  return neighbours;
}
