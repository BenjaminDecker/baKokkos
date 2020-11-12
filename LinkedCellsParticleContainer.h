//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once

#include "LinkedCell.h"
#include "SimulationConfig.h"

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
  std::optional<std::string> vtkFilename;

  LinkedCellsParticleContainer() = default;

  /**
   * @brief Initialises particles
   * @param parser stores particle information from a .yaml file
   */
  explicit LinkedCellsParticleContainer(std::vector<Particle> &particles, SimulationConfig &config);
  void iterateCalculatePositions(double deltaT);
  void iterateCalculateForces();
  void iterateCalculateVelocities(double deltaT);
  void writeVTKFile(int iteration, int maxIterations, const std::string &fileName) const;

 private:
  [[nodiscard]] int getIndexOf(int x, int y, int z) const;
  [[nodiscard]] std::array<int, 3> getCoordinates(int cellNumber) const;
  [[nodiscard]] std::vector<int> getNeighbourCellNumbers(int cellNumber);
};
