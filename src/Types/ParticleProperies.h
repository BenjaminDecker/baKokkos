//
// Created by Benjamin Decker on 26.11.20.
//

#pragma once

#include <Kokkos_Core.hpp>

/**
 * @brief A map from particle IDs to various particle properties.
 *
 * The map can be accessed concurrently from inside of a Kokkos::parallel_for().
 */
class ParticleProperties {
 public:

  float mass; /**< Particle Mass */

  /// Default constructor
  KOKKOS_INLINE_FUNCTION
  ParticleProperties() : mass(0.0) {}

  KOKKOS_INLINE_FUNCTION
  explicit ParticleProperties(float mass) : mass(mass) {}
};
