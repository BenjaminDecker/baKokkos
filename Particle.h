//
// Created by ffbde on 16/10/2020.
//

#ifndef BAKOKKOS_PARTICLE_H
#define BAKOKKOS_PARTICLE_H

#include <array>

class Particle {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
};


#endif //BAKOKKOS_PARTICLE_H
