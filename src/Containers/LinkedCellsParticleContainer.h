//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once

#include <array>
#include "Particle.h"
#include "../SimulationConfig/SimulationConfig.h"
#include "Cell.h"

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
 * A view of "cells" that each contain another view of type Cell which contains the particles the particles of that cell
 *
 * @see Cell
 */
typedef Kokkos::View<Cell *, SharedSpace> CellsViewType;

// TODO
constexpr enum BoundaryCondition {
  none, periodic, reflecting
} condition(periodic);

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
  CellsViewType cells; /**< Contains the linked cells that make up the simulation space */
  Kokkos::View<int *[27]> neighbours;
  Kokkos::View<int *> periodicTargetCellNumbers;
  Coord3D boxMin;
  Coord3D boxMax;
  int numCellsX;
  int numCellsY;
  int numCellsZ;
  int numCells;
  int iteration;
  const SimulationConfig config;

  std::array<Kokkos::View<int*>, 8> c08baseCells;

//  LinkedCellsParticleContainer() = default;

  /**
   * @brief Initialises particles
   * @param parser stores particle information from a .yaml file
   */
  explicit LinkedCellsParticleContainer(const SimulationConfig &config);
  void addParticle(const Particle &particle) const;
  [[nodiscard]] std::vector<Particle> getParticles() const;
  void doIteration();

 private:
  void calculatePositions() const;
  void calculateForces() const;
  void calculateForcesNewton3() const;
  void calculateVelocities() const;
  void moveParticles() const;
  [[nodiscard]] KOKKOS_FUNCTION int getCellNumberFromRelativeCellCoordinates(int x, int y, int z) const;
  [[nodiscard]] std::array<int, 3> getRelativeCellCoordinates(int cellNumber) const;
  [[nodiscard]] std::vector<int> getNeighbourCellNumbers(int cellNumber) const;
  [[nodiscard]] int getCorrectCellNumber(const Particle &particle) const;
  [[nodiscard]] int getCellColor(int cellNumber) const;
  void writeVTKFile() const;
};
