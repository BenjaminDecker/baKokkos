//
// Created by ffbde on 16/10/2020.
//

#include "Particle.h"

Particle::Particle(double x, double y, double z) {
    position = { x, y, z };
}

Particle::Particle(std::array<double, 3> position) {
    this->position = position;
}
