#include <iostream>
#include <Kokkos_Core.hpp>
#include "ParticlePropertiesLibrary.h"
#include "ParticleContainer.h"

int main (int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr double deltaT = 1.0;
        ParticlePropertiesLibrary particlePropertiesLibrary;
        ParticleContainer container;

        particlePropertiesLibrary.addParticleType(0, 1.0, 1.0, 1.0);

        //initialize a 100x100x100 cube of particles
        int idCounter = 0;
        for (int x = 0; x < 10; ++x) {
            for (int y = 0; y < 10; ++y) {
                for (int z = 0; z < 10; ++z) {
                    container.particles.emplace_back(idCounter++, 0,
                                                     Coord3D{static_cast<double>(x), static_cast<double>(y),
                                                             static_cast<double>(z)});
                }
            }
        }

        Kokkos::parallel_for("calculatePositions", container.particles.size(), KOKKOS_LAMBDA(int i) {
            auto particle = container.particles.at(i);
            particle.position = particle.position + particle.velocity * deltaT;
        });

        double epsilon = particlePropertiesLibrary.getEpsilon(0);
        double sigma = particlePropertiesLibrary.getSigma(0);

        for (int i = 0; i < container.particles.size(); ++i) {
            Particle &particle = container.particles.at(i);
            Coord3D &totalForce = particle.force;
            Coord3D &position = particle.position;
            std::cout << particle.force << std::endl;
            Kokkos::parallel_reduce("calculateForcesReduce", container.particles.size(),
                                    KOKKOS_LAMBDA(int n, Coord3D &totalForce) {
                                        auto otherParticle = container.particles.at(n);
                                        if (particle != otherParticle) {
                                            auto distance = position.distanceTo(otherParticle.position);
                                            auto distanceValue = distance.absoluteValue();
                                            auto distancePow6 = std::pow(distanceValue, 6);
                                            auto sigmaPow6 = std::pow(sigma, 6);

                                            auto forceValue = 24 * epsilon * sigmaPow6 *
                                                              ((distancePow6 - 2 * sigmaPow6) /
                                                               (distanceValue * distancePow6 *
                                                                distancePow6));  // https://www.ableitungsrechner.net/#expr=4%2A%CE%B5%28%28%CF%83%2Fr%29%5E12-%28%CF%83%2Fr%29%5E6%29&diffvar=r

                                            auto force = (distance / distanceValue) * forceValue;
                                            totalForce += force;
                                        }
                                    }, totalForce);
            std::cout << particle.force << std::endl << std::endl;
        }

        Kokkos::parallel_for("calculateVelocities", container.particles.size(), KOKKOS_LAMBDA(int i) {

        });
    }
    Kokkos::finalize();
    return 0;
}
