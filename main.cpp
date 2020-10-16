#include <iostream>
#include <Kokkos_Core.hpp>
#include "ParticleContainer.h"

int main (int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    double deltaT = 1.0;
    ParticleContainer container;

    //initialize a 100x100x100 cube of particles
    for (int x = 0; x < 100; ++x) {
        for (int y = 0; y < 100; ++y) {
            for (int z = 0; z < 100; ++z) {
                container.particles.emplace_back(Coord3D{ static_cast<double>(x), static_cast<double>(y), static_cast<double>(z) });
            }
        }
    }

    Kokkos::parallel_for("calculatePositions", container.particles.size(), KOKKOS_LAMBDA (int n) {
        auto particle = container.particles.at(n);
        particle.position = particle.position + particle.velocity * deltaT;
    });


    Kokkos::finalize();
    return 0;
}
