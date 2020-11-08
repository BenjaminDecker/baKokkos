//
// Created by Benjamin Decker on 22.10.20.
//

#pragma once

#include <Kokkos_Core.hpp>

/**
 * @brief A 3D coordinate struct designed to be used inside a Kokkos::View.
 *
 * The ParticleContainer uses instances of Kokkos::View<Coord3D> to save information about particles.
 * All functions that might be called from inside a Kokkos::parallel_for() or a Kokkos::parallel_reduce() must be
 * annotated with KOKKOS_INLINE_FUNCTION. Structs used as a reducer in a Kokkos::parallel_for need to have volatile
 * and non-volatile versions of the += operator defined.
 *
 * @see ParticleContainer
 */
struct Coord3D {
  double x, y, z;

  /// The default contructor must be explicitly declared to have the KOKKOS_INLINE_FUNCTION annotation.
  KOKKOS_INLINE_FUNCTION
  Coord3D() : x(0), y(0), z(0) {}

  /// The default copy contructor must be explicitly declared to have the KOKKOS_INLINE_FUNCTION annotation.
  KOKKOS_INLINE_FUNCTION
  Coord3D(const Coord3D &rhs) = default;

  /// The default destructor must be explicitly declared to have the KOKKOS_INLINE_FUNCTION annotation.
  KOKKOS_INLINE_FUNCTION
  ~Coord3D() = default;

  /// Convenient component-wise constructor.
  KOKKOS_INLINE_FUNCTION
  Coord3D(double x, double y, double z) : x(x), y(y), z(z) {}

  /// Coordinates of the distance vector between two Corrd3D objects.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D distanceTo(const Coord3D &other) const {
    return Coord3D(other.x - x, other.y - y, other.z - z);
  }

  /// Distance from the coordinate origin.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  double absoluteValue() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  /// Component-wise addition of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator+(const Coord3D &rhs) const {
    return Coord3D{x + rhs.x, y + rhs.y, z + rhs.z};
  }

  /// Component-wise multiplication of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator*(double rhs) const {
    return Coord3D{x * rhs, y * rhs, z * rhs};
  }

  /// Component-wise division of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator/(double rhs) const {
    return Coord3D{x / rhs, y / rhs, z / rhs};
  }

  /// Component-wise equality check of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  bool operator==(const Coord3D &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  /// Component-wise *= operator of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D &operator*=(double rhs) {
    x = x * rhs;
    y = y * rhs;
    z = z * rhs;
    return *this;
  }

  /// Component-wise non-volatile += operator of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D &operator+=(const Coord3D &rhs) {
    x = x + rhs.x;
    y = y + rhs.y;
    z = z + rhs.z;
    return *this;
  }

  /// Component-wise volatile += operator of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile Coord3D &rhs) volatile {
    x = x + rhs.x;
    y = y + rhs.y;
    z = z + rhs.z;
  }
};

/**
 * Operator definition for convenient printing of a Coord3D object to a Stream.
 * @example
 * Coord3D coord();
 * std::cout << coord << std::endl;
 */
KOKKOS_INLINE_FUNCTION
std::ostream &operator<<(std::ostream &stream, const Coord3D &obj) {
  stream << "( " << obj.x << ", " << obj.y << ", " << obj.z << " )";
  return stream;
}

namespace Kokkos {
template<>
/// Required to enable using Coord3D with Kokkos parallel_reduce
struct reduction_identity<Coord3D> {
  KOKKOS_FORCEINLINE_FUNCTION static Coord3D sum() {
    return Coord3D();
  }
};
}
