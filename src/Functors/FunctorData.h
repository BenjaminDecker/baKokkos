//
// Created by Benjamin Decker on 04.05.21.
//

#pragma once

#include <Kokkos_UnorderedMap.hpp>
#include "../Simulation.h"
#include "../BoundaryCondition.h"

/**
 * A class that saves information about the simulation for the functors to use. The functors cannot access data from the
 * simulation class directly if the code is compiled for CUDA. The FunctorData contains all the information used by all
 * functors combined so there is a bit of redundancy in the data that is copied to the functors, but the performance
 * impact is negligible and the code is a lot cleaner this way.
 */
class FunctorData {
 public:
  FunctorData(const Kokkos::View<Coord3D **> &positions,
              const Kokkos::View<Coord3D **> &velocities,
              const Kokkos::View<Coord3D **> &forces,
              const Kokkos::View<Coord3D **> &old_forces,
              const Kokkos::View<int **> &particle_i_ds,
              const Kokkos::View<int **> &type_i_ds,
              const Kokkos::View<int *> &cell_sizes,
              const Kokkos::View<int> &capacity,
              const Kokkos::View<bool *> &has_moved,
              const Kokkos::UnorderedMap<int, ParticleProperties> &particle_properties,
              const Coord3D &box_min,
              const Kokkos::View<Coord3D *> &bottom_left_corners,
              const int num_cellsX,
              const int num_cellsY,
              const int num_cellsZ,
              const int num_cells_total,
              const float cutoff,
              const float delta_t,
              const Kokkos::View<bool *> &is_halo,
              const Kokkos::View<int *[13][2]> &c_08_pairs,
              const Kokkos::View<int *> &periodic_target_cell_numbers,
              const BoundaryCondition &boundary_condition,
              const Kokkos::View<bool> &move_was_successful)
      : positions(positions),
        velocities(velocities),
        forces(forces),
        oldForces(old_forces),
        particleIDs(particle_i_ds),
        typeIDs(type_i_ds),
        cellSizes(cell_sizes),
        capacity(capacity),
        hasMoved(has_moved),
        particleProperties(particle_properties),
        boxMin(box_min),
        bottomLeftCorners(bottom_left_corners),
        numCells{num_cellsX, num_cellsY, num_cellsZ},
        numCellsTotal(num_cells_total),
        cutoff(cutoff),
        deltaT(delta_t),
        isHalo(is_halo),
        c08Pairs(c_08_pairs),
        periodicTargetCellNumbers(periodic_target_cell_numbers),
        boundaryCondition(boundary_condition),
        moveWasSuccessful(move_was_successful) {}

  /**
   * A 2-dimensional view that saves the positions of each particle for each cell. The first index specifies the cell,
   * the second index specifies the particle.
   */
  const Kokkos::View<Coord3D**> positions;

  /**
   * A 2-dimensional view that saves the velocities of each particle for each cell. The first index specifies the cell,
   * the second index specifies the particle.
   */
  const Kokkos::View<Coord3D**> velocities;

  /**
    * A 2-dimensional view that saves the forces of each particle for each cell. The first index specifies the cell,
    * the second index specifies the particle.
    */
  const Kokkos::View<Coord3D**> forces;

  /**
   * A 2-dimensional view that saves the old forces of each particle for each cell. The first index specifies the cell,
   * the second index specifies the particle.
   */
  const Kokkos::View<Coord3D**> oldForces;

  /**
   * A 2-dimensional view that saves the particleIDs of each particle for each cell. The first index specifies the cell,
   * the second index specifies the particle.
   */
  const Kokkos::View<int**> particleIDs;

  /**
   * A 2-dimensional view that saves the typeIDs of each particle for each cell. The first index specifies the cell,
   * the second index specifies the particle.
   */
  const Kokkos::View<int**> typeIDs;

  /// A view that saves the amount of particles for each cell.
  const Kokkos::View<int*> cellSizes;

  /// A view that saves the capacity of all cells. All cells always have the same capacity.
  const Kokkos::View<int> capacity;

  /// A view that saves whether or not a particle has moved outside of a cell in the last time step.
  const Kokkos::View<bool*> hasMoved;

  /// A mapping from particle IDs to various particle properties.
  const Kokkos::UnorderedMap<int, ParticleProperties> particleProperties;

  /**
   * The coordinate of the bottom left front corner of the simulation cube. Together with the numCells variable, this
   * is used to calculate whether or not a particle moved out of its cell.
   */
  const Coord3D boxMin;

  /// A view that saves the bottom left corner coordinates for each cell.
  const Kokkos::View<Coord3D *> bottomLeftCorners;

  /**
   * The amount of cells in each spacial direction in the simulation. Together with the boxMin variable, this
   * is used to calculate whether or not a particle moved out of its cell.
   */
  const int numCells[3];

  /**
   * The total amount of cells in the simulation
   */
  const int numCellsTotal;

  /// Maximum distance between two particles for which the force calculation can not be neglected to increase performance
  const float cutoff;

  /// Length of one time step of the simulation
  const float deltaT;

  /// A view that saves for each cell whether it is a halo cell or not.
  const Kokkos::View<bool*> isHalo;

  /**
   * A view that saves all 13 cell pairs of the c08 base step for all base cells. The first index specifies the cell
   * number of the base step, the second index specifies the pair of this base step, and the third index specifies the
   * cell of this pair.
   */
  const Kokkos::View<int *[13][2]> c08Pairs;

  /**
   * A view that contains the cell number of the periodic target cell on the opposite side of the simulation space for
   * each halo cell. If a particle moves into a halo cell, it has to be moved into its periodic target cell instead.
   * The size of the view is equal to the total amount of cells to enable easy access via cell number
   * indices. This means that there are redundant entries for the non-halo cells.
   */
  const Kokkos::View<int *> periodicTargetCellNumbers;

  /// The boundary condition that is used.
  const BoundaryCondition boundaryCondition;

  /**
   * A view that saves whether or not the move step was successful. Before each move step this is set to true. When a
   * particle moves inside of a cell that has reached max capacity, the moveWasSuccessful property is set to false. If
   * the move was not successful, the cells have to be resized.
   */
  const Kokkos::View<bool> moveWasSuccessful;
};
