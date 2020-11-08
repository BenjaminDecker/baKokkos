//
// Created by Benjamin Decker on 08.11.20.
//

#pragma once

#include "ParticleContainer.h"

class LinkedCellsParticleContainer : public ParticleContainer {
  /// Creates a Particle from the particle information in device memory with the specified id.
  [[nodiscard]] Particle getParticle(int id) const override;

  /// Inserts the information stored in a Particle into device memory with the specified id.
  void insertParticle(const Particle &particle, int id) const override;
};



