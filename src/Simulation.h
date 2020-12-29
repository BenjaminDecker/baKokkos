//
// Created by Benjamin Decker on 03/11/2020.
//

#pragma once

#include "SimulationConfig/SimulationConfig.h"
#include "Types/Cell.h"
#include "Types/ParticleProperies.h"
#include <spdlog//spdlog.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <vector>

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

constexpr enum BoundaryCondition {
  none, periodic, reflecting
} boundaryCondition(reflecting);

/**
 * @brief This class controls the simulation. The simulation space is partitioned into cells, which contain the particles.
 *
 * The cells are cubes with a side length equal to the cutoff value. Forces acting between particles that are further
 * apart from another than the cutoff radius are very small and can therefore be neglected to improve performance. By
 * only calculating forces between neighbouring cells, many of these unnecessary calculations are automatically ignored.
 * This layout improves performance if the simulation contains many particles, as the time complexity for the force
 * calculation grows with only O(n) instead of O(n^2).
 *
 * @see Particle, Coord3D
 */
class Simulation {
 public:
  const SimulationConfig config; /**< Configuration for the simulation */
  CellsViewType cells; /**< Contains the linked cells that make up the simulation space */
  Kokkos::View<int *[27]> neighbours; /**< Contains the cell numbers of all neighbours for each cell */
  Kokkos::UnorderedMap<int, ParticleProperties>
      particleProperties; /**< Map from particle type to particle properties */
  Coord3D boxMin; /**< Lower-Left-Front corner of the simulation space */
  Coord3D boxMax; /**< Upper-Right-Back corner of the simulation space */
  int numCellsX; /**< Number of cells in the x-direction */
  int numCellsY; /**< Number of cells in the y-direction */
  int numCellsZ; /**< Number of cells in the z-direction */
  int numCells; /**< Total number of cells */
  int iteration; /**< The current iteration */

  /**
   * Contains the cell number of the periodic target cell on the opposite side of the simulation space for each halo
   * cell. The size of the view is equal to the total amount of cells to enable easy access via cell number indices. The
   * periodic target cell numbers for non-halo cells are equal to the cell number index.
   */
  Kokkos::View<int *> periodicTargetCellNumbers;

  /// Contains 8 views of c08-base-cell cell numbers for each of the 8 different colors of the c08 cell coloring
  std::array<Kokkos::View<int *>, 8> c08baseCells;

  //TODO change to 2-element int
  /// Contains all cell pairs of the c08 base step for each of the c08 base cell numbers
  Kokkos::View<std::pair<int, int> *[13]> c08Pairs;

  /// Initializes the simulation by creating all views and adding all particles
  explicit Simulation(const SimulationConfig &config);

  /// Starts the simulation loop
  void start() {
    spdlog::info("Running Simulation...");
    Kokkos::Timer timer;

    //Iteration loop
    for (; iteration < config.iterations; ++iteration) {
      if (iteration % 1000 == 0) {
        spdlog::info("Iteration: {:0" + std::to_string(std::to_string(config.iterations).length()) + "d}", iteration);
      }
      calculatePositions();
      calculateForcesNewton3();
      calculateVelocities();
      moveParticles();
      if (config.vtk && iteration % config.vtk.value().second == 0) {
        writeVTKFile(config.vtk.value().first);
      }
    }

    const double time = timer.seconds();
    spdlog::info("Finished simulating. Time: " + std::to_string(time) + " seconds.");
  }

  /// This method finds the correct cell and inserts the given particle into it
  void addParticle(const Particle &particle) const;

  /// This method adds all particles from all cells into one std::vector and returns it
  [[nodiscard]] std::vector<Particle> getParticles() const;

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
  void writeVTKFile(const std::string &fileBaseName) const;
};
