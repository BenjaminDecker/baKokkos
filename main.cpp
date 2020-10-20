#include <iostream>
#include <Kokkos_Core.hpp>
#include "ParticlePropertiesLibrary.h"
#include "ParticleContainer.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int iterations = 1000;
    constexpr double deltaT = 1.0;
    constexpr double cubeSideLength = 2;
    constexpr double epsilon = 1;
    constexpr double sigma = 1;
    constexpr double mass = 1;

    /*
    ParticlePropertiesLibrary particlePropertiesLibrary;
    particlePropertiesLibrary.addParticleType(0, 1.0, 1.0, 1.0);
     */


    //Creates the particle container and initializes a cube of particles
    ParticleContainer container(cubeSideLength);

    //Iteration loop
    for (int timeStep = 0; timeStep < iterations; ++timeStep) {

      //Calculate positions
      Kokkos::parallel_for("calculatePositions", container.size, KOKKOS_LAMBDA(int i) {
        Coord3D &position = container.positions(i);
        position = position + container.velocities(i) * deltaT;
      });

      //Calculate forces
      //Iterate over every particle
      Kokkos::parallel_for("forceIteration", container.size, KOKKOS_LAMBDA(int i) {
        Coord3D position = container.positions(i);
        Coord3D totalForce = Coord3D();

        //Calculate every particle's force contribution
        Kokkos::parallel_reduce("forceReduction", container.size, KOKKOS_LAMBDA(int n, Coord3D &totalForce) {

          //Skip the calculation if the two particles are the same
          if (n == i) {
            return;
          }

          Coord3D distance = position.distanceTo(container.positions(n));
          double distanceValue = distance.absoluteValue();
          double distanceValuePow6 = std::pow(distanceValue, 6);
          double sigmaPow6 = std::pow(sigma, 6);

          // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r
          double forceValue = (24 * epsilon * sigmaPow6 * (distanceValuePow6 - 2 * sigmaPow6)) /
              (distanceValue * distanceValuePow6 * distanceValuePow6);

          Coord3D force = ((distance / distanceValue) * forceValue);
          totalForce += force;
        }, totalForce);

        //Assign the new total force for this time step
        container.forces(i) = totalForce;
      });

      //Calculate the new velocities
      Kokkos::parallel_for("calculateVelocities", container.size, KOKKOS_LAMBDA(int i) {
        Coord3D deltaVelocity = (container.forces(i) / mass) * deltaT;
        container.velocities(i) += deltaVelocity;
      });

      //Test prints
      for (int i = 0; i < container.size; ++i) {
        std::cout << container.positions(i);
      }
      std::cout << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
