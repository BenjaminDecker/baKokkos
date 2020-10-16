//
// Created by ffbde on 16/10/2020.
//

#include "ParticleContainer.h"

void ParticleContainer::addParticle(double x, double y, double z) {
    particles.emplace_back(x, y, z);
}

void ParticleContainer::addParticle(Particle p) {
    particles.emplace_back(p.position);
}