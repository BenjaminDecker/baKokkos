#include <fstream>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>
#include "ParticlePropertiesLibrary.h"
#include "Coord3D.h"
#include "ParticleContainer.h"

void writeVTKFile(unsigned int iteration, unsigned int iterationCount, const ParticleContainer &container);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {

    constexpr int iterations = 10000000;
    constexpr double deltaT = 0.000002;
    constexpr double cubeSideLength = 2;
    constexpr double epsilon = 1;
    constexpr double sigma = 1;
    constexpr double mass = 1;

    /*
    ParticlePropertiesLibrary particlePropertiesLibrary;
    particlePropertiesLibrary.addParticleType(0, 1.0, 1.0, 1.0);
     */

    std::cout << "Initializing particles..." << std::endl;
    Kokkos::Timer timer1;

    //Creates the particle container and initializes a cube of particles
    ParticleContainer container(cubeSideLength);
    double time1 = timer1.seconds();
    std::cout << "Finished initializing " << container.size << " particles." << std::endl << "Time: " << time1
              << " seconds" << std::endl << std::endl;

    std::cout << "Running Simulation..." << std::endl;
    Kokkos::Timer timer2;

    //Iteration loop
    for (int timeStep = 0; timeStep < iterations; ++timeStep) {

      //Calculate positions
      Kokkos::parallel_for("calculatePositions", container.size, KOKKOS_LAMBDA(int i) {
        container.positions(i) += container.velocities(i) * deltaT;
      });

      typedef Kokkos::TeamPolicy<> team_policy;
      typedef Kokkos::TeamPolicy<>::member_type member_type;

      //Calculate forces
      Kokkos::parallel_for("calculateForces",
                           team_policy(container.size, Kokkos::AUTO()),
                           KOKKOS_LAMBDA(const member_type &teamMember) {
                             int id_1 = teamMember.league_rank();
                             Coord3D force = Coord3D();
                             Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, container.size),
                                                     [=](int id_2, Coord3D &force) {
                                                       if (id_1 == id_2) {
                                                         return;
                                                       }

                                                       Coord3D distance =
                                                           container.positions(id_1).distanceTo(container.positions(id_2));
                                                       double distanceValue = distance.absoluteValue();
                                                       double distanceValuePow6 = std::pow(distanceValue, 6);
                                                       double sigmaPow6 = std::pow(sigma, 6);

                                                       // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
                                                       double forceValue = -(24 * epsilon * sigmaPow6
                                                           * (distanceValuePow6 - 2 * sigmaPow6)) /
                                                           (distanceValue * distanceValuePow6 * distanceValuePow6);

                                                       force += (distance * (forceValue / distanceValue));
                                                     }, force);
                             container.forces(id_1) = force;
                           });


      //Calculate the new velocities
      Kokkos::parallel_for("calculateVelocities", container.size, KOKKOS_LAMBDA(int i) {
        Coord3D deltaVelocity = (container.forces(i) / mass) * deltaT;
        container.velocities(i) += deltaVelocity;
      });

      if (timeStep % 10000 == 0) {
        writeVTKFile(timeStep, iterations, container);
      }

//      //Test prints
//      for (int i = 0; i < container.size; ++i) {
//        std::cout << container.forces(i);
//      }
//      std::cout << std::endl;
    }
    double time2 = timer2.seconds();
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
    auto &coord = container.positions(i);
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  vtkFile << "POINT_DATA " << numParticles << std::endl;
  // print velocities
  vtkFile << "VECTORS velocities double" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    auto &coord = container.velocities(i);
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  // print Forces
  vtkFile << "VECTORS forces double" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    auto &coord = container.forces(i);
    vtkFile << coord.x << " " << coord.y << " " << coord.z << std::endl;
  }
  vtkFile << std::endl;

  // print TypeIDs
  vtkFile << "SCALARS typeIds int" << std::endl;
  vtkFile << "LOOKUP_TABLE default" << std::endl;
  for (int i = 0; i < numParticles; ++i) {
    vtkFile << 0 << std::endl;
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