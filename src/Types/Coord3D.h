//
// Created by Benjamin Decker on 22.10.20.
//

#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>

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
class Coord3D {
 public:
  float x, y, z;

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
  Coord3D(float x, float y, float z) : x(x), y(y), z(z) {}

  /// Coordinates of the distance vector between two Corrd3D objects.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D distanceTo(const Coord3D &other) const {
    return Coord3D(other.x - x, other.y - y, other.z - z);
  }

  /// Distance from the coordinate origin.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  float absoluteValue() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  /// Component-wise addition of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator+(const Coord3D &rhs) const {
    return Coord3D{x + rhs.x, y + rhs.y, z + rhs.z};
  }

  /// Component-wise subtraction of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator-(const Coord3D &rhs) const {
    return Coord3D{x - rhs.x, y - rhs.y, z - rhs.z};
  }

  /// Scalar multiplication of a Coord3D with a float.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator*(float rhs) const {
    return Coord3D{x * rhs, y * rhs, z * rhs};
  }

  /// Scalar division of a Coord3D with a float.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator/(float rhs) const {
    return Coord3D{x / rhs, y / rhs, z / rhs};
  }

  /// Component-wise equality check of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  bool operator==(const Coord3D &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  /// Scalar *= of a Coord3D with a float.
  KOKKOS_INLINE_FUNCTION
  Coord3D &operator*=(float rhs) {
    x = x * rhs;
    y = y * rhs;
    z = z * rhs;
    return *this;
  }

  /// Scalar /= of a Coord3D with a float.
  KOKKOS_INLINE_FUNCTION
  Coord3D &operator/=(float rhs) {
    x = x / rhs;
    y = y / rhs;
    z = z / rhs;
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

  /// Returns a point that was rotated from this points position along a line with the specified roll, pitch and yaw values in radians that goes through the specified point
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
  Coord3D rotateRollPitchYaw(float roll, float pitch, float yaw, const Coord3D &rotationPoint) const {
    // https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw

    float sinRoll = std::sin(roll);
    float cosRoll = std::cos(roll);
    float sinPitch = std::sin(pitch);
    float cosPitch = std::cos(pitch);
    float sinYaw = std::sin(yaw);
    float cosYaw = std::cos(yaw);

    Coord3D newPoint = Coord3D(x, y, z) + rotationPoint * (-1);
    float newX = newPoint.x, newY = newPoint.y, newZ = newPoint.z;

    // Rx
    {
      float tmpY = newY, tmpZ = newZ;
      newY = tmpY * cosRoll - tmpZ * sinRoll;
      newZ = tmpY * sinRoll + tmpZ * cosRoll;
    }

    // Ry
    {
      float tmpX = newX, tmpZ = newZ;
      newX = tmpX * cosPitch + tmpZ * sinPitch;
      newZ = -tmpX * sinPitch + tmpZ * cosPitch;
    }

    // Rz
    {
      float tmpX = newX, tmpY = newY;
      newX = tmpX * cosYaw - tmpY * sinYaw;
      newY = tmpX * sinYaw + tmpY * cosYaw;
    }

    return Coord3D(newX, newY, newZ) + rotationPoint;
  }

};

/**
 * Operator definition for convenient printing of a Coord3D object to a Stream.
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
