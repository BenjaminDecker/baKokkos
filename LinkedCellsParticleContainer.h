//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once

#include <array>
#include "LinkedCell.h"
#include "SimulationConfig.h"

#ifdef KOKKOS_ENABLE_CUDA
/**
 * The memory space that is used for views that contain other views. The outer view has to be accessible by both the cpu
 * and the parallel device that the program is compiled for.
 * The program was compiled to use Cude as its parallel device so CudaUVMSpace has to used for the outer view.
 */
#define SharedSpace Kokkos::CudaUVMSpace
#endif
#ifndef SharedSpace
/**
 * The memory space that is used for views that contain other views. The outer view has to be accessible by both the cpu
 * and the parallel device that the program is compiled for.
 * The program was not compiled to use any special device so the DefaultExecutionSpace can be used for the outer view.
 */
#define SharedSpace Kokkos::DefaultExecutionSpace
#endif

/**
 * A view that saves a size and a capacity for each of the cells inside a ContainerViewType for dynamic resizing of each of the cells
 *
 * @see ContainerViewType
 */
using SizesAndCapacitiesType = Kokkos::View<int *[2], SharedSpace>;

/**
 * A view of "particles" that each consist of a 3 dimensional coordinate for each of its 4 properties (position, force, oldForce, velocity)
 */
using CellViewType = Kokkos::View<Coord3D *[4]>;

/**
 * A view of "cells" that each contain another view of type CellViewType which contains the particles the particles of that cell
 *
 * @see CellViewType
 */
using ContainerViewType = Kokkos::View<CellViewType *, SharedSpace>;

enum ParticleIndices {
  position = 0,
  velocity = 1,
  force = 2,
  oldForce = 3
};

/**
 * @brief Saves particles inside of cells that make up the simulation space
 *
 * The cells are cubes with a side length equal to the cutoff value. If it can be assumed that the force calculations
 * of particles that are further away from another than the cutoff value are so small that they can be neglected, only
 * the forces between particles from the same or neighboring cells have to be calculated. ALl other particle pairs
 * at least a distance equal to cutoff away from another. This automatically filters out many of the unnecessary force
 * calculations.
 * This layout improves performance if the simulation contains more than a handful of particles, as the time complexity
 * for the force calculation grows with only O(n).
 *
 * @see Particle, Coord3D
 */
class LinkedCellsParticleContainer {
 public:
  ContainerViewType cells; /**< Contains the linked cells that make up the simulation space */
  SizesAndCapacitiesType sizesAndCapacites;
  Kokkos::View<int *[27]> neighbours;
  Coord3D boxMin;
  Coord3D boxMax;
  int numCellsX{};
  int numCellsY{};
  int numCellsZ{};
  int numCells{};
  double cutoff{};
  std::optional<std::string> vtkFilename;

  LinkedCellsParticleContainer() = default;

  /**
   * @brief Initialises particles
   * @param parser stores particle information from a .yaml file
   */
  explicit LinkedCellsParticleContainer(std::vector<Particle> &particles, SimulationConfig &config);
  void addParticle(const Particle &particle);
  [[nodiscard]] std::vector<Particle> getParticles() const;
  void iterateCalculatePositions(double deltaT) const;
  void iterateCalculateForces() const;
  void iterateCalculateVelocities(double deltaT) const;
  void writeVTKFile(int iteration, int maxIterations, const std::string &fileName) const;

 private:
  [[nodiscard]] int getIndexOf(int x, int y, int z) const;
  [[nodiscard]] std::array<int, 3> getCoordinates(int cellNumber) const;
  [[nodiscard]] std::vector<int> getNeighbourCellNumbers(int cellNumber);
  void resizeCellCapacity(int cellNumber, int factor) const;
};
