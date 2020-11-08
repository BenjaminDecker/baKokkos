//
// Created by ffbde on 03/11/2020.
//

#include "Simulation.h"
Simulation::Simulation(const SimulationConfig &config) : config(config) {
  spdlog::info("Initializing particles...");
  Kokkos::Timer timer;
  container = ParticleContainer(YamlParser(config.yamlFileName));
  const double time = timer.seconds();
  spdlog::info("Finished initializing " + std::to_string(container.size) + " particles. Time: "
                   + std::to_string(time) + " seconds.");
}

void Simulation::start() const {
  spdlog::info("Running Simulation...");
  Kokkos::Timer timer;

  //Iteration loop
  for (int iteration = 0; iteration < config.iterations; ++iteration) {

    //Calculate positions
    Kokkos::parallel_for("calculatePositions", container.size, KOKKOS_LAMBDA(int i) {
      container.positions(i) +=
          container.velocities(i) * config.deltaT
              + container.forces(i) * ((config.deltaT * config.deltaT) / (2 * mass));
    });

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = Kokkos::TeamPolicy<>::member_type;

    //Calculate forces
    Kokkos::parallel_for("calculateForces",
                         team_policy(container.size, Kokkos::AUTO),
                         KOKKOS_LAMBDA(const member_type &teamMember) {
                           const int id_1 = teamMember.league_rank();
                           Coord3D force = Coord3D();
                           Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, container.size),
                                                   [=](const int id_2, Coord3D &force) {

                                                     if (id_1 == id_2) {
                                                       return;
                                                     }

                                                     const Coord3D distance =
                                                         container.positions(id_1).distanceTo(container.positions(id_2));
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
                                                   }, force);
                           container.oldForces(id_1) = container.forces(id_1);
                           container.forces(id_1) = force;
                         });


    //Calculate the new velocities
    Kokkos::parallel_for("calculateVelocities", container.size, KOKKOS_LAMBDA(int i) {
      container.velocities(i) += (container.forces(i) + container.oldForces(i)) * (config.deltaT / (2 * mass));
    });

    if (config.vtkOutput) {
      if (iteration % config.vtkWriteFrequency == 0) {
        writeVTKFile(iteration);
      }
    }

    if (iteration % 1000 == 0) {
      spdlog::info("Iteration: {:0" + std::to_string(std::to_string(config.iterations).length()) + "d}", iteration);
    }
  }

  const double time = timer.seconds();
  spdlog::info("Finished simulating. Time: " + std::to_string(time) + " seconds.");
}
void Simulation::writeVTKFile(int iteration) const {
  std::string fileBaseName("baKokkos");
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(config.iterations).length();
  unsigned int numParticles = container.size;
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
  vtkFile << "POINTS " << numParticles << " double" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    auto coord = container.getParticle(i).position;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  vtkFile << "POINT_DATA " << numParticles << std::endl;
  // print velocities
  vtkFile << "VECTORS velocities double" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    auto coord = container.getParticle(i).velocity;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  // print Forces
  vtkFile << "VECTORS forces double" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    auto coord = container.getParticle(i).force;
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  // print TypeIDs
  vtkFile << "SCALARS typeIds int" << std::endl;
  vtkFile << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    vtkFile << container.getParticle(i).typeID << std::endl;
  }
  vtkFile << std::endl;

  // print TypeIDs
  vtkFile << "SCALARS particleIds int" << std::endl;
  vtkFile << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    vtkFile << i << std::endl;
  }
  vtkFile << std::endl;
  vtkFile.close();
}
