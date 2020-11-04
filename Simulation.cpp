//
// Created by ffbde on 03/11/2020.
//

#include "Simulation.h"
Simulation::Simulation(int argc, char *argv[]) {
  cxxopts::Options options("baKokkos");
  options.add_options("Non-mandatory")
      ("help", "Display this message")
      ("iterations", "Number of iterations to simulate", cxxopts::value<int>()->default_value("100000"))
      ("deltaT", "Length of one time step of the simulation",cxxopts::value<double>()->default_value("0.000002"))
      ("vtk-filename", "Basename for all VTK output files", cxxopts::value<std::string>())
      ("vtk-write-frequency", "Number of iterations after which a VTK file is written", cxxopts::value<int>()->default_value("10000"))
      ("yaml-filename", "Path to a .yaml input file", cxxopts::value<std::string>());
  auto result = options.parse(argc, argv);
  iterations = result["iterations"].as<int>();
  deltaT = result["deltaT"].as<double>();
  if (result.count("vtk-filename") > 0) {
    vtkOutput = true;
    vtkFileName = result["vtk-filename"].as<std::string>();
    vtkWriteFrequency = result["vtk-write-frequency"].as<int>();
  }
  if (result.count("yaml-filename") > 0) {
    yamlInput = true;
    yamlFileName = result["yaml-filename"].as<std::string>();
  }
  std::cout << "Initializing particles..." << std::endl;
  Kokkos::Timer timer;
  container = ParticleContainer(YamlParser(yamlFileName));
  const double time = timer.seconds();
  std::cout << "Finished initializing " << container.size << " particles." << std::endl << "Time: " << time
            << " seconds" << std::endl << std::endl;
}


void Simulation::start() const {
  std::cout << "Running Simulation..." << std::endl;
  Kokkos::Timer timer;

  //Iteration loop
  for (int iteration = 0; iteration < iterations; ++iteration) {

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
      container.velocities(i) += (container.forces(i) + container.oldForces(i)) * (deltaT / (2 * mass));
    });

    if (vtkOutput) {
      if (iteration % vtkWriteFrequency == 0) {
        writeVTKFile(iteration);
      }
    }

    if (iteration % 1000 == 0) {
      std::cout << iteration << std::endl;
    }
  }

  const double time = timer.seconds();
  std::cout << "Finished simulating" << std::endl << "Time: " << time << " seconds" << std::endl << std::endl;

}
void Simulation::writeVTKFile(int iteration) const {
  std::string fileBaseName("baKokkos");
  std::ostringstream strstr;
  auto maxNumDigits = std::to_string(iterations).length();
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
