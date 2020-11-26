//
// Created by Benjamin Decker on 26.11.20.
//

#pragma once

#include <Kokkos_Core.hpp>

class ParticleProperties {
 public:
  double mass;

  KOKKOS_INLINE_FUNCTION
  ParticleProperties() : mass(0.0) {}

  KOKKOS_INLINE_FUNCTION
  explicit ParticleProperties (double mass) : mass(mass) {}
};
