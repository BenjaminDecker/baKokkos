//
// Created by ffbde on 16/10/2020.
//

#include "Particle.h"

Particle::Particle(double x, double y, double z) {
    position = { x, y, z };
}

Particle::Particle(Coord3D position) {
    this->position = position;
}
