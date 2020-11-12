//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <fstream>
#include <Kokkos_Vector.hpp>
#include "ParticleContainer.h"
#include "Coord3D.h"
#include "Particle.h"
#include "YamlParser.h"
#include "LinkedCell.h"

/**
 * @brief Divides the simulation domain into cells, which contain particles.
 *
 * The cells are cubes with a sidelength of the cutoff property. This means that every particle in one cell only needs
 * to calculate the contributing force to and from other particles within the same cell and particles inside neighboring
 * cells. This automatically filters out many of the force calculations of particle pairs that are further away from
 * another than the cutoff distance and can therefore be neglected.
 * This layout improves performance if the simulation contains more than a handful of particles, as the time complexity
 * for the force calculation grows with only O(n).
 *
 * @see Particle, Coord3D
 */
class LinkedCellsParticleContainer {
 public:
  std::vector<LinkedCell> cells;
  int numCellsX{};
  int numCellsY{};
  int numCellsZ{};
  double cutoff{};
  Coord3D boxMin;
  Coord3D boxMax;
  std::string vtkFilename;

  LinkedCellsParticleContainer() = default;

  /**
   * @brief Initialises particles
   * @param parser stores particle information from a .yaml file
   */
  explicit LinkedCellsParticleContainer(const YamlParser &parser)
      : cutoff(parser.cutoff), boxMin(parser.boxMin), boxMax(parser.boxMax) {
    spdlog::info("Initializing particles...");
    Kokkos::Timer timer;
    Coord3D boxSize = boxMax - boxMin;
    numCellsX = static_cast<int>(boxSize.x / cutoff);
    numCellsY = static_cast<int>(boxSize.y / cutoff);
    numCellsZ = static_cast<int>(boxSize.z / cutoff);
    int numCells = numCellsX * numCellsY * numCellsZ;
    cells.resize(numCells);

    std::vector<Particle> particles;
    for (auto &cuboid : parser.particleCuboids) {
      cuboid.getParticles(particles);
    }
    for (auto &sphere : parser.particleSpheres) {
      sphere.getParticles(particles);
    }

    for (auto &particle : particles) {
      Coord3D cellPosition = (particle.position - boxMin) / cutoff;
      int cellNumber = getIndexOf(static_cast<int>(cellPosition.x),
                                  static_cast<int>(cellPosition.y),
                                  static_cast<int>(cellPosition.z));
      cells[cellNumber].addParticle(particle);
    }
    const double time = timer.seconds();
    spdlog::info("Finished initializing " + std::to_string(particles.size()) + " particles. Time: "
                     + std::to_string(time) + " seconds.");
  }

  void iterateCalculatePositions(double deltaT) {
    //TODO get from particlePropertiesLibrary
    constexpr double mass = 1;
    for (auto &cell : cells) {
      Kokkos::parallel_for(cell.size, KOKKOS_LAMBDA(int i) {
        cell.positions(i) += cell.velocities(i) * deltaT + cell.forces(i) * ((deltaT * deltaT) / (2 * mass));
      });
    }
  }

  void iterateCalculateForces() {
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
                                 const Coord3D distance = cells[0].positions(id_1).distanceTo(cells[cell].positions(id_2));
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

  void iterateCalculateVelocities(double deltaT) {
    //TODO get from particlePropertiesLibrary
    constexpr int mass = 1;
    for (auto &cell : cells) {
      Kokkos::parallel_for(cell.size, KOKKOS_LAMBDA(int i) {
        cell.velocities(i) += (cell.forces(i) + cell.oldForces(i)) * (deltaT / (2 * mass));
      });
    }
  }

  void writeVTKFile(int iteration, int maxIterations, const std::string &fileName) const {

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
    vtkFile << "POINTS " << particles.size()<< " double" << std::endl;
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

 private:

  [[nodiscard]] int getIndexOf(int x, int y, int z) const {
    return z * numCellsX * numCellsY + y * numCellsX + x;
  }

  [[nodiscard]] std::array<int, 3> getCoordinates(int cellNumber) const {
    int z = cellNumber / (numCellsX * numCellsY);
    cellNumber -= z * (numCellsX * numCellsY);
    int y = cellNumber / numCellsX;
    cellNumber -= y * numCellsX;
    return {cellNumber, y, z};
  }

  [[nodiscard]] std::vector<int> getNeighbourCellNumbers(int cellNumber) {
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
  }

  [[nodiscard]] inline int calculateCell(const Coord3D &position) const {
    if (position.x < boxMin.x || position.y < boxMin.y || position.z < boxMin.z) {
      return -1;
    }
    int cellX = static_cast<int>((position.x - boxMin.x) / cutoff);
    int cellY = static_cast<int>((position.y - boxMin.y) / cutoff);
    int cellZ = static_cast<int>((position.z - boxMin.z) / cutoff);
    return getIndexOf(cellX, cellY, cellZ);
  }

};
