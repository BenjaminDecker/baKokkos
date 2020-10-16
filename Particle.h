//
// Created by ffbde on 16/10/2020.
//

#ifndef BAKOKKOS_PARTICLE_H
#define BAKOKKOS_PARTICLE_H

#include <array>

class Particle {
public:
    Particle(double x, double y, double z);
    Particle(std::array<double, 3> position);

    std::array<double, 3> position;
    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
};


#endif //BAKOKKOS_PARTICLE_H
