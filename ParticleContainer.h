//
// Created by ffbde on 16/10/2020.
//

#ifndef BAKOKKOS_PARTICLECONTAINER_H
#define BAKOKKOS_PARTICLECONTAINER_H

#include <vector>
#include "Particle.h"

class ParticleContainer {
private:
    std::vector<Particle> particles;
public:
    void addParticle(double x, double y, double z);
    void addParticle(Particle p);
};

#endif //BAKOKKOS_PARTICLECONTAINER_H
