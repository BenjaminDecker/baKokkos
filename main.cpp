#include <fstream>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>
#include "ParticlePropertiesLibrary.h"
#include "Coord3D.h"
#include "ParticleContainer.h"
#include "Simulation.h"



void writeVTKFile(unsigned int iteration, unsigned int iterationCount, const ParticleContainer &container);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Simulation simulation = Simulation(argc, argv);

    /*
    ParticlePropertiesLibrary particlePropertiesLibrary;
    particlePropertiesLibrary.addParticleType(0, 1.0, 1.0, 1.0);
     */

    std::cout << "Initializing particles..." << std::endl;
    Kokkos::Timer timer1;

    //Creates the particle container and initializes a cube of particles
    ParticleContainer container(YamlParser(result["yaml-filename"].as<std::string>()));
    const double time1 = timer1.seconds();
    std::cout << "Finished initializing " << container.size << " particles." << std::endl << "Time: " << time1
              << " seconds" << std::endl << std::endl;

    std::cout << "Running Simulation..." << std::endl;
    Kokkos::Timer timer2;

    //Iteration loop
    for (int timeStep = 0; timeStep < iterations; ++timeStep) {

      //Calculate positions
      Kokkos::parallel_for("calculatePositions", container.size, KOKKOS_LAMBDA(int i) {
        container.positions(i) +=
            container.velocities(i) * deltaT + container.forces(i) * ((deltaT * deltaT) / (2 * mass));
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

                                                       Coord3D distance =
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
        container.velocities(i) += (container.forces(i) + container.oldForces(i)) * (deltaT / (2 * mass));
      });

      if (timeStep % 10000 == 0) {
        writeVTKFile(timeStep, iterations, container);
      }
      if (timeStep % 100 == 0) {
        std::cout << timeStep << std::endl;
      }

//      //Test prints
//      for (int i = 0; i < container.size; ++i) {
//        std::cout << container.forces(i);
//      }
//      std::cout << std::endl;
    }
    const double time2 = timer2.seconds();
    std::cout << "Finished simulating" << std::endl << "Time: " << time2 << " seconds" << std::endl << std::endl;

  }
  Kokkos::finalize();
  return 0;
}

void writeVTKFile(unsigned int iteration, unsigned int iterationCount, const ParticleContainer &container) {
  std::string fileBaseName("baKokkos");
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(iterationCount).length();
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