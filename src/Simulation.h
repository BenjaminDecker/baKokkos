//
// Created by Benjamin Decker on 03/11/2020.
//

#pragma once

#include "SimulationConfig/SimulationConfig.h"
#include "Types/ParticleProperies.h"
#include <spdlog//spdlog.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_DualView.hpp>
#include <vector>

/// A Struct to define the boundary condition that is used in the simulation
enum BoundaryCondition {
  none, periodic, reflecting
};

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
  /// The boundary condition that is used in the simulation (Currently this is still hard-coded)
  const BoundaryCondition boundaryCondition = none;

  /// Simulation configuration data
  const SimulationConfig config;

  /**
   * A 2-dimensional view that saves all particle positions in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  Kokkos::View<Coord3D**> positions;

  /**
   * A 2-dimensional view that saves all particle forces in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  Kokkos::View<Coord3D**> forces;

  /**
   * A 2-dimensional view that saves all old particle forces in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  Kokkos::View<Coord3D**> oldForces;

  /**
   * A 2-dimensional view that saves all particle velocities in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  Kokkos::View<Coord3D**> velocities;

  /**
   * A 2-dimensional view that saves all particle typeIDs in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  Kokkos::View<int**> typeIDs;

  /**
   * A 2-dimensional view that saves all particleIDs in all cells. The first index specifies the cell, the second
   * index specifies the particle.
   */
  Kokkos::View<int**> particleIDs;

  /// A view that saves for each cell whether it is a halo cell or not.
  Kokkos::View<bool*> isHalo;

  /// A view that saves the bottom left corner coordinates for each cell.
  Kokkos::View<Coord3D *> bottomLeftCorners;

  /// A view that saves the amount of particles inside each cell.
  Kokkos::View<int *> cellSizes;

  /// A view that saves the capacity of all cells. All cells always have the same capacity.
  Kokkos::View<int> capacity;

  /// A view that saves whether or not a particle has moved outside of a cell in the last time step.
  Kokkos::View<bool*> hasMoved;

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
   * A view that saves whether or not the move step was successful. Before each move step this is set to true. When a
   * particle moves inside of a cell that has reached max capacity, the moveWasSuccessful property is set to false. If
   * the move was not successful, the cells have to be resized.
   */
  Kokkos::View<bool> moveWasSuccessful;


  /**
   * A view that contains the cell number of the periodic target cell on the opposite side of the simulation space for
   * each halo cell. If a particle moves into a halo cell, it has to be moved into its periodic target cell instead.
   * The size of the view is equal to the total amount of cells to enable easy access via cell number
   * indices. This means that there are redundant entries for the non-halo cells.
   */
  Kokkos::View<int *> periodicTargetCellNumbers;

  /// Contains 8 views of c08-base-cell cell numbers for each of the 8 different colors of the c08 cell coloring
  std::array<Kokkos::View<int *>, 8> c08baseCells;

  std::array<Kokkos::View<int *>, 27> moveParticlesBaseCells;

  /**
   * A view that saves all 13 cell pairs of the c08 base step for all base cells. The first index specifies the cell
   * number of the base step, the second index specifies the pair of this base step, and the third index specifies the
   * cell of this pair.
   */
  Kokkos::View<int *[13][2]> c08Pairs;

  /// Initializes the simulation by creating all views and adding all particles
  explicit Simulation(SimulationConfig config);

  /// Starts the simulation loop
  void start();

  void addParticles(const std::vector<Particle> &particles);

  /// Returns a std::vector of all particles inside the simulation
  [[nodiscard]] std::vector<Particle> getParticles() const;

  /**
   * Calculates the pairwise forces on particle pairs and adds the calculated force to both particles. This is more
   * efficient than only calculating the received force for every particle. The implementation uses c08-base-cells
   */
  void calculateForcesNewton3() const;

  /**
   * Calculates the velocities and positions of all particles after deltaT seconds based on their current velocities,
   * positions and acting forces
   */
  void calculateVelocitiesAndPositions() const;

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


  [[nodiscard]] int getCorrectCellNumber(const Coord3D &position) const;

  /**
   * Returns a number from 0 to 7, representing the color of the cell with the given cell number for the c08 base cell
   * color scheme
   */
  [[nodiscard]] std::pair<int, int> getCellColors(int cellNumber) const;

  /**
   * Writes a new vtk file containing information about the position, velocity, experienced force, typeID and particleID
   * of all particles in the simulation
   */
  void writeVTKFile(const std::string &fileBaseName) const;

  /**
   * The simulation is initializes. Because of the gcc cuda compiler, the initialization of the simulation has to happen
   * outside of the simulation constructor.
   */
  void initializeSimulation();
};
