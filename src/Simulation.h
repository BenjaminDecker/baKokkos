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
#include <Kokkos_DualView.hpp>
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

typedef Kokkos::View<Coord3D *> InnerCoordViewType;
typedef Kokkos::View<int *> InnerIntViewType;
typedef Kokkos::View<InnerCoordViewType *, SharedSpace> CoordViewType;
typedef Kokkos::View<InnerIntViewType *, SharedSpace> IntViewType;

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
  //CellsViewType cells; /**< Contains the linked cells that make up the simulation space */

  CoordViewType positions;
  CoordViewType forces;
  CoordViewType oldForces;
  CoordViewType velocities;
  IntViewType typeIDs;
  IntViewType particleIDs;
  Kokkos::View<bool*> isHalo;
  decltype(Kokkos::create_mirror_view(isHalo)) h_isHalo;
  Kokkos::View<Coord3D *> bottomLeftCorners;
  Kokkos::DualView<int *> cellSizes;
  std::vector<int> cellCapacities;

  Kokkos::View<int *[27]> neighbours; /**< Contains the cell numbers of all neighbours for each cell */
  Kokkos::UnorderedMap<int, ParticleProperties> particleProperties; /**< Map of particle properties */
  Coord3D boxMin; /**< Lower-Left-Front corner of the simulation space */
  Coord3D boxMax; /**< Upper-Right-Back corner of the simulation space */
  int numCellsX; /**< Number of cells in the x-direction */
  int numCellsY; /**< Number of cells in the y-direction */
  int numCellsZ; /**< Number of cells in the z-direction */
  int numCells; /**< Total number of cells */
  int iteration; /**< The current iteration */

  /**
   * Contains the cell number of each periodic target cell on the opposite side of the simulation space for each halo
   * cell. The size of the view is equal to the total amount of cells to enable easy access via cell number indices.
   * This means that there are redundant entries for the non-halo cells.
   */
  Kokkos::View<int *> periodicTargetCellNumbers;

  /// Contains 8 views of c08-base-cell cell numbers for each of the 8 different colors of the c08 cell coloring
  std::array<Kokkos::View<int *>, 8> c08baseCells;

  /// Contains all cell pairs of the c08 base step for each of the c08 base cell numbers
  Kokkos::View<int *[13][2]> c08Pairs;

  /// Initializes the simulation by creating all views and adding all particles
  explicit Simulation(SimulationConfig config);

  /// Starts the simulation loop
  void start();

  /// Inserts the given particle into the correct cell
  void addParticle(const Particle &particle);

  [[nodiscard]] std::vector<Particle> getParticles(int cellNumber) const;

  /// Returns a std::vector of all particles inside the simulation
  [[nodiscard]] std::vector<Particle> getParticles() const;

  void removeParticle(int cellNumber, int index);

  /// Calculate the positions of all particles after deltaT seconds based on their current positions and velocities
  void calculatePositions() const;

  /**
   * Calculates the force acting on every particle based on the lj 6-12 potential and the particles positions. For each
   * particle in a cell, the force acting on that particle from every other particle in the same cell and in all
   * neighbouring cells is calculated and added together, plus the global force acting on every particle.
   */
  void calculateForces() const;

  /**
   * Calculates the pairwise forces on particle pairs and adds the calculated force to both particles. This is more
   * efficient than only calculating the received force for every particle. The implementation uses c08-base-cells
   */
  void calculateForcesNewton3() const;

  /**
   * Calculates the velocities of all particles after deltaT seconds based on their current velocities and acting forces
   */
  void calculateVelocities() const;

  /**
   * Checks for every particle if the particle is still saved in the correct cell, given by its coordinates, and moves
   * the particle into the correct cell if necessary
   */
  void moveParticles();

  /**
   * Returns the cellNumber of the cell at the specified position in a cell grid with dimensions given by the numCells
   * member variables and with the first cell at position (0,0,0)
   */
  [[nodiscard]] int getCellNumberFromRelativeCellCoordinates(int x, int y, int z) const;

  /**
   * Returns the position of the cell with the specified cellNumber in a cell grid with dimensions given by the numCells
   * member variables and with the first cell at position (0,0,0)
   */
  [[nodiscard]] std::array<int, 3> getRelativeCellCoordinates(int cellNumber) const;

  /**
   * Returns a vector of all cellNumbers in a 3x3x3 cube around the cell with the specified cell number. Only existing
   * cellNumbers are returned, so the returned vector contains less elements for halo cells.
   */
  [[nodiscard]] std::vector<int> getNeighbourCellNumbers(int cellNumber) const;

  /**
   * Returns the correct cell number of a particle, given by its position
   */
  [[nodiscard]] int getCorrectCellNumber(const Particle &particle) const;

  /**
   * Returns a number from 0 to 7, representing the color of the cell with the given cell number for the c08 base cell
   * color scheme
   */
  [[nodiscard]] int getCellColor(int cellNumber) const;

  /**
   * Writes a new vtk file containing information about the position, velocity, experienced force, typeID and particleID
   * of all particles in the simulation
   */
  void writeVTKFile(const std::string &fileBaseName) const;

  void initializeSimulation();
};
