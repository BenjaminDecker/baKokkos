//
// Created by Benjamin Decker on 29.12.20.
//

#include "Simulation.h"
#include <iomanip>
#include <fstream>
#include <utility>
#include "Functors/CalculateForces.h"
#include "Functors/CalculateVelocitiesAndPositions.h"
#include "Functors/MoveParticles.h"
//#include <cuda_profiler_api.h>

KOKKOS_INLINE_FUNCTION
void getRelativeCellCoordinatesDevice(int cellNumber, int cellsX, int cellsY, int &x, int &y, int &z) {
  z = cellNumber / (cellsX * cellsY);
  cellNumber -= z * (cellsX * cellsY);
  y = cellNumber / cellsX;
  x = cellNumber - y * cellsX;
}

Simulation::Simulation(SimulationConfig config) : config(std::move(config)), iteration(0) {
  Kokkos::Timer timer;
  initializeSimulation();
}

void Simulation::start() {
  spdlog::info("Running Simulation...");
  Kokkos::Timer timer;
//  cudaProfilerStart();
  for (; iteration < config.iterations; ++iteration) {
    calculateForcesNewton3();
    calculateVelocitiesAndPositions();
    moveParticles();
    if (config.vtk && iteration % config.vtk->second == 0) {
      writeVTKFile(config.vtk.value().first);
    }
  }
//  cudaProfilerStop();
}

void Simulation::addParticles(const std::vector<Particle> &particles) {
  auto h_positions = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), positions);
  auto h_velocities = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), velocities);
  auto h_forces = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), forces);
  auto h_oldForces = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), oldForces);
  auto h_particleIDs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), particleIDs);
  auto h_typeIDs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), typeIDs);
  auto h_cellSizes = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), cellSizes);
  auto h_capacity = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), capacity);

  for(const auto &particle : particles) {
    const int cellNumber = getCorrectCellNumber(particle);
    if (cellNumber < 0 || numCells <= cellNumber) {
      std::cout
          << "Particles outside of the simulation space cuboid cannot be added to any cell of the simulation."
          << std::endl;
      exit(1);
    }
    if (h_cellSizes(cellNumber) == h_capacity()) {
      h_capacity() *= 2;
      Kokkos::resize(h_positions, numCells, h_capacity());
      Kokkos::resize(h_velocities, numCells, h_capacity());
      Kokkos::resize(h_forces, numCells, h_capacity());
      Kokkos::resize(h_oldForces, numCells, h_capacity());
      Kokkos::resize(h_particleIDs, numCells, h_capacity());
      Kokkos::resize(h_typeIDs, numCells, h_capacity());
    }
    h_positions(cellNumber, h_cellSizes(cellNumber)) = particle.position;
    h_velocities(cellNumber, h_cellSizes(cellNumber)) = particle.velocity;
    h_forces(cellNumber, h_cellSizes(cellNumber)) = particle.force;
    h_oldForces(cellNumber, h_cellSizes(cellNumber)) = particle.oldForce;
    h_particleIDs(cellNumber, h_cellSizes(cellNumber)) = particle.particleID;
    h_typeIDs(cellNumber, h_cellSizes(cellNumber)) = particle.typeID;
    ++h_cellSizes(cellNumber);
  }
  Kokkos::resize(positions, numCells, h_capacity());
  Kokkos::resize(velocities, numCells, h_capacity());
  Kokkos::resize(forces, numCells, h_capacity());
  Kokkos::resize(oldForces, numCells, h_capacity());
  Kokkos::resize(particleIDs, numCells, h_capacity());
  Kokkos::resize(typeIDs, numCells, h_capacity());

  Kokkos::deep_copy(positions, h_positions);
  Kokkos::deep_copy(velocities, h_velocities);
  Kokkos::deep_copy(forces, h_forces);
  Kokkos::deep_copy(oldForces, h_oldForces);
  Kokkos::deep_copy(particleIDs, h_particleIDs);
  Kokkos::deep_copy(typeIDs, h_typeIDs);
  Kokkos::deep_copy(capacity, h_capacity);
  Kokkos::deep_copy(cellSizes, h_cellSizes);
}

std::vector<Particle> Simulation::getParticles() const {
  std::vector<Particle> particles;
  const auto h_positions = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), positions);
  const auto h_velocities = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), velocities);
  const auto h_forces = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), forces);
  const auto h_oldForces = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), oldForces);
  const auto h_particleIDs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), particleIDs);
  const auto h_typeIDs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), typeIDs);
  const auto h_cellSizes = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), cellSizes);

  for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
    for (int i = 0; i < h_cellSizes(cellNumber); ++i) {
      particles.emplace_back(h_particleIDs(cellNumber, i),
                             h_typeIDs(cellNumber, i),
                             h_positions(cellNumber, i),
                             h_forces(cellNumber, i),
                             h_velocities(cellNumber, i),
                             h_oldForces(cellNumber, i)
      );
    }
  }
  return particles;
}

void Simulation::calculateForcesNewton3() const {
  // Save oldForces and initialize new forces
  Kokkos::deep_copy(oldForces, forces);
  Kokkos::deep_copy(forces, config.globalForce);

  Kokkos::fence();

  for (int color = 0; color < 8; ++color) {
    const auto colorCells = c08baseCells[color];
    Kokkos::parallel_for(
        "calculateForcesForColor: " + std::to_string(color) + "  Iteration: " + std::to_string(iteration),
        Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, colorCells.size()),
        CalculateForces(*this, colorCells)
    );
    Kokkos::fence();
  }
}

void Simulation::calculateVelocitiesAndPositions() const {
  Kokkos::deep_copy(hasMoved, false);
  Kokkos::fence();
  Kokkos::parallel_for(
      "calculateVelocitiesAndPosition  Iteration: " + std::to_string(iteration),
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, numCells),
      CalculateVelocitiesAndPositions(*this)
  );
  Kokkos::fence();
}

void Simulation::moveParticles() {
  for (int color = 0; color < 27; ++color) {
    Kokkos::deep_copy(moveWasSuccessful, true);
    bool h_moveWasSuccessful = false;
    const auto colorCells = moveParticlesBaseCells[color];
    while (!h_moveWasSuccessful) {
      Kokkos::parallel_for("moveParticles",
                           Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, colorCells.size()),
                           MoveParticles(*this, colorCells)
      );
      Kokkos::fence();
      Kokkos::deep_copy(h_moveWasSuccessful, moveWasSuccessful);
      if (!h_moveWasSuccessful) {
        int h_capacity;
        Kokkos::deep_copy(h_capacity, capacity);
        h_capacity *= 2;
        Kokkos::deep_copy(capacity, h_capacity);
        Kokkos::resize(positions, numCells, h_capacity);
        Kokkos::resize(velocities, numCells, h_capacity);
        Kokkos::resize(forces, numCells, h_capacity);
        Kokkos::resize(oldForces, numCells, h_capacity);
        Kokkos::resize(particleIDs, numCells, h_capacity);
        Kokkos::resize(typeIDs, numCells, h_capacity);
        Kokkos::deep_copy(moveWasSuccessful, true);
      }
    }
  }
}

void Simulation::writeVTKFile(const std::string &fileBaseName) const {
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(config.iterations).length();
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

  vtkFile << "# vtk DataFile Version 2.0" << "\n";
  vtkFile << "Timestep" << "\n";
  vtkFile << "ASCII" << "\n";

  // print positions
  vtkFile << "DATASET STRUCTURED_GRID" << "\n";
  vtkFile << "DIMENSIONS 1 1 1" << "\n";
  vtkFile << "POINTS " << particles.size() << " float" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].position;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }
  vtkFile << "\n";

  vtkFile << "POINT_DATA " << particles.size() << "\n";
  // print velocities
  vtkFile << "VECTORS velocities float" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].velocity;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }
  vtkFile << "\n";

  // print Forces
  vtkFile << "VECTORS forces float" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    auto coord = particles[i].force;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << "\n";
  }
  vtkFile << "\n";

  // print TypeIDs
  vtkFile << "SCALARS typeIds int" << "\n";
  vtkFile << "LOOKUP_TABLE default" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << particles[i].typeID << "\n";
  }
  vtkFile << "\n";

  // print ParticleIDs
  vtkFile << "SCALARS particleIds int" << "\n";
  vtkFile << "LOOKUP_TABLE default" << "\n";
  for (int i = 0; i < particles.size(); ++i) {
    vtkFile << particles[i].particleID << "\n";
  }
  vtkFile << std::endl;
  vtkFile.close();
}

void Simulation::initializeSimulation() {
  std::cout << "Using the following simulation configuration:\n\n" << config << "\n" << std::endl;
  spdlog::info("Initializing particles...");
  Kokkos::Timer timer;

  capacity = Kokkos::View<int>("capacity");
  Kokkos::deep_copy(capacity, 1);

  std::vector<Particle> particles;

  /*
   * All particles are extracted from the particle groups and saved inside an std::vector. At the same time the
   * Kokkos::UnorderedMap particleProperties is constructed and filled with all new particleTypes. The size of the map
   * is set to the amount of particleGroups as this is the maximum possible amount of different particleTypes for the
   * simulation. In reality, this number is a lot smaller because many particleGroups will have the same particleType.
   */
  {
    particleProperties = Kokkos::UnorderedMap<int, ParticleProperties>(config.particleGroups.size());
    const auto particlePropertiesCopy = particleProperties;
    for (const auto &particleGroup : config.particleGroups) {
      const int typeID = particleGroup->typeID;
      const ParticleProperties pp(particleGroup->particleMass);
      Kokkos::parallel_for("add particle properties", 1, KOKKOS_LAMBDA(int i) {
        if (!particlePropertiesCopy.exists(typeID)) {
          particlePropertiesCopy.insert(typeID, pp);
        }
      });
      Kokkos::fence();
      const auto newParticles = particleGroup->getParticles(particles.size());
      for (const auto &particle : newParticles) {
        particles.push_back(particle);
      }
    }
  }

  /*
   * If values for the size of the simulation space are given, they are used. Otherwise size values are created by
   * starting with one cell that is positioned at the mean position of all particles. Afterwards, more cells are added
   * in all 3 spacial directions until the cells together contain all particles. The cells are added in all 3 directions
   * separately, so the simulation space is not necessarily a cube, but always a cuboid.
   */
  {
    if (config.box) {
      boxMin = config.box.value().first;
      boxMax = config.box.value().second;
    } else {
      float lowestX = particles[0].position.x;
      float lowestY = particles[0].position.y;
      float lowestZ = particles[0].position.z;
      float highestX = particles[0].position.x;
      float highestY = particles[0].position.y;
      float highestZ = particles[0].position.z;
      Coord3D midPoint = Coord3D();
      for (auto &particle : particles) {
        lowestX = std::min(lowestX, particle.position.x);
        lowestY = std::min(lowestY, particle.position.y);
        lowestZ = std::min(lowestZ, particle.position.z);
        highestX = std::max(highestX, particle.position.x);
        highestY = std::max(highestY, particle.position.y);
        highestZ = std::max(highestZ, particle.position.z);
        midPoint += particle.position;
      }
      midPoint /= particles.size();
      boxMin = midPoint - Coord3D(config.cutoff / 2, config.cutoff / 2, config.cutoff / 2);
      boxMax = midPoint + Coord3D(config.cutoff / 2, config.cutoff / 2, config.cutoff / 2);
      while (lowestX < boxMin.x) {
        boxMin = boxMin - Coord3D(config.cutoff, 0, 0);
      }
      while (boxMax.x + config.cutoff < highestX) {
        boxMax = boxMax + Coord3D(config.cutoff, 0, 0);
      }
      while (lowestY < boxMin.y) {
        boxMin = boxMin - Coord3D(0, config.cutoff, 0);
      }
      while (boxMax.y + config.cutoff < highestY) {
        boxMax = boxMax + Coord3D(0, config.cutoff, 0);
      }
      while (lowestZ < boxMin.z) {
        boxMin = boxMin - Coord3D(0, 0, config.cutoff);
      }
      while (boxMax.z + config.cutoff < highestZ) {
        boxMax = boxMax + Coord3D(0, 0, config.cutoff);
      }
    }

    // A one cell wide layer of empty cells is added around the simulation space to act as halo cells in the simulation.
    boxMin = boxMin - Coord3D(config.cutoff, config.cutoff, config.cutoff) * 2;
    boxMax = boxMax + Coord3D(config.cutoff, config.cutoff, config.cutoff) * 2;

    // The total size of the simulation, and with it the number of cells is calculated.
    const Coord3D boxSize = boxMax - boxMin + Coord3D(config.cutoff, config.cutoff, config.cutoff);
    numCellsX = static_cast<int>((boxSize.x + 0.5 * config.cutoff) / config.cutoff);
    numCellsY = static_cast<int>((boxSize.y + 0.5 * config.cutoff) / config.cutoff);
    numCellsZ = static_cast<int>((boxSize.z + 0.5 * config.cutoff) / config.cutoff);
    numCells = numCellsX * numCellsY * numCellsZ;
  }

  /*
   * The cells and periodicTargetCellNumbers views are initialized to the correct size. The periodicTargetCellNumbers
   * view will only be used if periodic boundary conditions are used in the simulation, but is still always created. It
   * has a size of numCells, although it only needs to store information for the halo cells. Both actions do not create
   * too much computational overhead, while simplifying the code a lot.
   * The cells view has to be created with the Kokkos::WithoutInitializing flag because it will contain other views.
   */
  {
    cellSizes = Kokkos::View<int *>("cellSizes", numCells);
    const auto h_cellSizes = Kokkos::create_mirror_view(cellSizes);
    hasMoved = Kokkos::View<bool *>("hasMoved", numCells);
    const auto h_hasMoved = Kokkos::create_mirror_view(hasMoved);
    for (int i = 0; i < numCells; ++i) {
      h_cellSizes(i) = 0;
      h_hasMoved(i) = false;
    }
    Kokkos::deep_copy(cellSizes, h_cellSizes);
    Kokkos::deep_copy(hasMoved, h_hasMoved);

    moveWasSuccessful = Kokkos::View<bool>("moveWasSuccessful");

    positions = Kokkos::View<Coord3D **>("positions", numCells, 1);
    forces = Kokkos::View<Coord3D **>("forces", numCells, 1);
    oldForces = Kokkos::View<Coord3D **>("oldForces", numCells, 1);
    velocities = Kokkos::View<Coord3D **>("velocities", numCells, 1);
    typeIDs = Kokkos::View<int **>("typeIDs", numCells, 1);
    particleIDs = Kokkos::View<int **>("particleIDs", numCells, 1);

    periodicTargetCellNumbers = Kokkos::View<int *>("periodicTargetCellNumbers", numCells);
    isHalo = Kokkos::View<bool *>("haloCells", numCells);
    bottomLeftCorners = Kokkos::View<Coord3D *>("bottomLeftCorners", numCells);
    auto h_periodicTargetCellNumbers = Kokkos::create_mirror_view(periodicTargetCellNumbers);
    auto h_isHalo = Kokkos::create_mirror_view(isHalo);
    auto h_bottomLeftCorners = Kokkos::create_mirror_view(bottomLeftCorners);

    // All necessary cells are created and saved in the cells view.
    for (int x = 0; x < numCellsX; ++x) {
      for (int y = 0; y < numCellsY; ++y) {
        for (int z = 0; z < numCellsZ; ++z) {
          //TODO maybe delete this
          const bool isHaloCell =
              x == 0 || x == numCellsX - 1 || y == 0 || y == numCellsY - 1 || z == 0 || z == numCellsZ - 1;
          const Coord3D bottomLeftCorner = boxMin + Coord3D(x, y, z) * config.cutoff;
          const int cellNumber = getCellNumberFromRelativeCellCoordinates(x, y, z);

          // The cells have to be saved via the new operator because they are views inside of views.
          //new(&cells[cellNumber]) Cell(1, isHaloCell, bottomLeftCorner);
          h_isHalo(cellNumber) = isHaloCell;
          h_bottomLeftCorners(cellNumber) = bottomLeftCorner;


          // For non-halo cells the periodicTargetCellNumber is equal to the cellNumber. For halo cells they are different.
          const int targetX = x == 0 ? numCellsX - 2 : x == numCellsX - 1 ? 1 : x;
          const int targetY = y == 0 ? numCellsY - 2 : y == numCellsY - 1 ? 1 : y;
          const int targetZ = z == 0 ? numCellsZ - 2 : z == numCellsZ - 1 ? 1 : z;
          const int periodicTargetCellNumber = getCellNumberFromRelativeCellCoordinates(targetX, targetY, targetZ);
          h_periodicTargetCellNumbers(cellNumber) = periodicTargetCellNumber;
        }
      }
    }
    Kokkos::deep_copy(periodicTargetCellNumbers, h_periodicTargetCellNumbers);
    Kokkos::deep_copy(isHalo, h_isHalo);
    Kokkos::deep_copy(bottomLeftCorners, h_bottomLeftCorners);
  }

  // For each cell, the neighbours view is filled with the cell numbers of its neighbours.
  {
    neighbours = Kokkos::View<int *[27]>("neighbours", numCells);
    const auto h_neighbours = Kokkos::create_mirror_view(neighbours);
    for (int i = 0; i < numCells; ++i) {
      std::vector<int> neighbourNumbers = getNeighbourCellNumbers(i);
      for (int k = 0; k < neighbourNumbers.size(); ++k) {
        h_neighbours(i, k) = neighbourNumbers[k];
      }
      for (int k = neighbourNumbers.size(); k < 27; ++k) {
        h_neighbours(i, k) = -1;
      }
    }
    Kokkos::deep_copy(neighbours, h_neighbours);
  }

  {
    std::vector<std::vector<int>> c08baseCellsVec;
    std::vector<std::vector<int>> moveParticlesBaseCellsVec;
    c08baseCellsVec.resize(8);
    moveParticlesBaseCellsVec.resize(27);
    for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
      const auto colors = getCellColors(cellNumber);
      c08baseCellsVec[colors.first].push_back(cellNumber);
      moveParticlesBaseCellsVec[colors.second].push_back(cellNumber);
    }

    for (int i = 0; i < 8; ++i) {
      const int size = c08baseCellsVec[i].size();
      c08baseCells[i] = Kokkos::View<int *>("c08baseCells " + std::to_string(i), size);
      const auto h_c08BaseCells = Kokkos::create_mirror_view(c08baseCells[i]);
      for (int k = 0; k < size; ++k) {
        h_c08BaseCells(k) = c08baseCellsVec[i][k];
      }
      Kokkos::deep_copy(c08baseCells[i], h_c08BaseCells);
    }
    for (int i = 0; i < 27; ++i) {
      const int size = moveParticlesBaseCellsVec[i].size();
      moveParticlesBaseCells[i] = Kokkos::View<int *>("moveParticlesBaseCells " + std::to_string(i), size);
      const auto h_moveParticlesBaseCells = Kokkos::create_mirror_view(moveParticlesBaseCells[i]);
      for (int k = 0; k < size; ++k) {
        h_moveParticlesBaseCells(k) = moveParticlesBaseCellsVec[i][k];
      }
      Kokkos::deep_copy(moveParticlesBaseCells[i], h_moveParticlesBaseCells);
    }
  }


  // For each c08 base cell, all cell pairs for force interactions are saved into the c08Pairs view.
  {
    c08Pairs = Kokkos::View<int *[13][2]>("c08Pairs", numCells);
    auto h_c08Pairs = Kokkos::create_mirror_view(c08Pairs);
    for (int cellNumber = 0; cellNumber < numCells; ++cellNumber) {
      auto coords = getRelativeCellCoordinates(cellNumber);
      int index = 0;
      // Iterate over every neighbour cell with a higher cell number.
      for (int x = coords[0] - 1; x < coords[0] + 2; ++x) {
        for (int y = coords[1] - 1; y < coords[1] + 2; ++y) {
          for (int z = coords[2] - 1; z < coords[2] + 2; ++z) {
            if (cellNumber >= getCellNumberFromRelativeCellCoordinates(x, y, z)) {
              continue;
            }
            Coord3D cellOne(coords[0], coords[1], coords[2]);
            Coord3D cellTwo(x, y, z);

            /*
             * If one of the 3 coordinates of the second cell is smaller than the coordinate of the first cell,
             * increment this coordinate for both cells.
             */
            if (x < coords[0]) {
              cellOne += Coord3D(1, 0, 0);
              cellTwo += Coord3D(1, 0, 0);
            }
            if (y < coords[1]) {
              cellOne += Coord3D(0, 1, 0);
              cellTwo += Coord3D(0, 1, 0);
            }
            if (z < coords[2]) {
              cellOne += Coord3D(0, 0, 1);
              cellTwo += Coord3D(0, 0, 1);
            }
            int cellNumberOne = getCellNumberFromRelativeCellCoordinates(cellOne.x, cellOne.y, cellOne.z);
            int cellNumberTwo = getCellNumberFromRelativeCellCoordinates(cellTwo.x, cellTwo.y, cellTwo.z);
            h_c08Pairs(cellNumber, index, 0) = cellNumberOne;
            h_c08Pairs(cellNumber, index++, 1) = cellNumberTwo;
          }
        }
      }
    }
    Kokkos::deep_copy(c08Pairs, h_c08Pairs);
  }
  Kokkos::fence();

  // After all cells are initialized, the particles are added
  addParticles(particles);
  Kokkos::fence();
  const float time = timer.seconds();
  spdlog::info("Finished initializing " + std::to_string(particles.size()) + " particles in " + std::to_string(numCells) + " cells. Time: "
                   + std::to_string(time) + " seconds.");
}

int Simulation::getCellNumberFromRelativeCellCoordinates(const int x, const int y, const int z) const {
  return z * numCellsX * numCellsY + y * numCellsX + x;
}

std::array<int, 3> Simulation::getRelativeCellCoordinates(int cellNumber) const {
  int z = cellNumber / (numCellsX * numCellsY);
  cellNumber -= z * (numCellsX * numCellsY);
  int y = cellNumber / numCellsX;
  cellNumber -= y * numCellsX;
  return {cellNumber, y, z};
}

std::vector<int> Simulation::getNeighbourCellNumbers(const int cellNumber) const {
  std::vector<int> neighbourNumbers;
  const auto coords = getRelativeCellCoordinates(cellNumber);
  for (int z = coords[2] - 1; z <= coords[2] + 1; ++z) {
    for (int y = coords[1] - 1; y <= coords[1] + 1; ++y) {
      for (int x = coords[0] - 1; x <= coords[0] + 1; ++x) {
        if (0 <= x && x < numCellsX && 0 <= y && y < numCellsY && 0 <= z && z < numCellsZ) {
          neighbourNumbers.push_back(getCellNumberFromRelativeCellCoordinates(x, y, z));
        }
      }
    }
  }
  return neighbourNumbers;
}

int Simulation::getCorrectCellNumber(const Particle &particle) const {
  return getCorrectCellNumber(particle.position);
}

int Simulation::getCorrectCellNumber(const Coord3D &position) const {
  const Coord3D cellPosition = (position - boxMin) / config.cutoff;
  return getCellNumberFromRelativeCellCoordinates(static_cast<int>(cellPosition.x),
                                                  static_cast<int>(cellPosition.y),
                                                  static_cast<int>(cellPosition.z));
}

std::pair<int,int> Simulation::getCellColors(const int cellNumber) const {
  const auto coords = getRelativeCellCoordinates(cellNumber);
  const int c08Color = (coords[0] % 2) * 1 + (coords[1] % 2) * 2 + (coords[2] % 2) * 4;
  const int moveParticlesColor = (coords[0] % 3) * 1 + (coords[1] % 3) * 3 + (coords[2] % 3) * 9;
  return {c08Color, moveParticlesColor};
}
