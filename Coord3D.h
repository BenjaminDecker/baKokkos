//
// Created by Benjamin Decker on 22.10.20.
//

#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>

struct Coord3D {
  double x, y, z;

  KOKKOS_INLINE_FUNCTION
  Coord3D() : x(0), y(0), z(0) {}

  KOKKOS_INLINE_FUNCTION
  Coord3D(const Coord3D &rhs) = default;

  KOKKOS_INLINE_FUNCTION
  Coord3D(double x, double y, double z) : x(x), y(y), z(z) {}

  KOKKOS_INLINE_FUNCTION
  ~Coord3D() = default;

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D distanceTo(const Coord3D &other) const {
    return Coord3D{x - other.x, y - other.y, z - other.z};
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  double absoluteValue() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  KOKKOS_INLINE_FUNCTION
  Coord3D operator+(const Coord3D &rhs) const {
    return Coord3D{x + rhs.x, y + rhs.y, z + rhs.z};
  }

  KOKKOS_INLINE_FUNCTION
  Coord3D operator*(double rhs) const {
    return Coord3D{x * rhs, y * rhs, z * rhs};
  }

  KOKKOS_INLINE_FUNCTION
  Coord3D operator/(double rhs) const {
    return Coord3D{x / rhs, y / rhs, z / rhs};
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const Coord3D &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  KOKKOS_INLINE_FUNCTION
  Coord3D &operator*=(double rhs) {
    x = x * rhs;
    y = y * rhs;
    z = z * rhs;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Coord3D &operator+=(const Coord3D &rhs) {
    x = x + rhs.x;
    y = y + rhs.y;
    z = z + rhs.z;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile Coord3D &rhs) volatile {
    x = x + rhs.x;
    y = y + rhs.y;
    z = z + rhs.z;
  }
};

KOKKOS_INLINE_FUNCTION
std::ostream &operator<<(std::ostream &stream, const Coord3D &obj) {
  stream << "( " << obj.x << ", " << obj.y << ", " << obj.z << " )";
  return stream;
}

namespace Kokkos {
template<>
struct reduction_identity<Coord3D> {
  KOKKOS_FORCEINLINE_FUNCTION static Coord3D sum() {
    return Coord3D();
  }
};
}
