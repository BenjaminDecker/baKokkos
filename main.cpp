#include <iostream>
#include <Kokkos_Core.hpp>
#include "ParticleContainer.h"

int main (int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    ParticleContainer container;

    //initialize a 100x100x100 cube of particles
    for (int x = 0; x < 100; ++x) {
        for (int y = 0; y < 100; ++y) {
            for (int z = 0; z < 100; ++z) {
                container.addParticle(x, y, z);
            }
        }
    }


    Kokkos::finalize();
    return 0;
}
