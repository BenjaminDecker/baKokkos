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
  double x, y, z;

  /// The default contructor must be explicitly declared to have the KOKKOS_INLINE_FUNCTION annotation.
  KOKKOS_INLINE_FUNCTION
  Coord3D() : x(0), y(0), z(0) {}

  /// The default copy contructor must be explicitly declared to have the KOKKOS_INLINE_FUNCTION annotation.
  Coord3D(const Coord3D &rhs) = default;

  /// The default destructor must be explicitly declared to have the KOKKOS_INLINE_FUNCTION annotation.
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

  /// Component-wise subtraction of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator-(const Coord3D &rhs) const {
    return Coord3D{x - rhs.x, y - rhs.y, z - rhs.z};
  }

  /// Scalar multiplication of a Coord3D with a double.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator*(double rhs) const {
    return Coord3D{x * rhs, y * rhs, z * rhs};
  }

  /// Scalar division of a Coord3D with a double.
  KOKKOS_INLINE_FUNCTION
  Coord3D operator/(double rhs) const {
    return Coord3D{x / rhs, y / rhs, z / rhs};
  }

  /// Component-wise equality check of two Coord3D's.
  KOKKOS_INLINE_FUNCTION
  bool operator==(const Coord3D &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  /// Scalar *= of a Coord3D with a double.
  KOKKOS_INLINE_FUNCTION
  Coord3D &operator*=(double rhs) {
    x = x * rhs;
    y = y * rhs;
    z = z * rhs;
    return *this;
  }

  /// Scalar /= of a Coord3D with a double.
  KOKKOS_INLINE_FUNCTION
  Coord3D &operator/=(double rhs) {
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
  Coord3D rotateRollPitchYaw(double roll, double pitch, double yaw, const Coord3D &rotationPoint) const {
    // https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw

    double sinRoll = std::sin(roll);
    double cosRoll = std::cos(roll);
    double sinPitch = std::sin(pitch);
    double cosPitch = std::cos(pitch);
    double sinYaw = std::sin(yaw);
    double cosYaw = std::cos(yaw);

    Coord3D newPoint = Coord3D(x, y, z) + rotationPoint * (-1);
    double newX = newPoint.x, newY = newPoint.y, newZ = newPoint.z;

    // Rx
    {
      double tmpY = newY, tmpZ = newZ;
      newY = tmpY * cosRoll - tmpZ * sinRoll;
      newZ = tmpY * sinRoll + tmpZ * cosRoll;
    }

    // Ry
    {
      double tmpX = newX, tmpZ = newZ;
      newX = tmpX * cosPitch + tmpZ * sinPitch;
      newZ = -tmpX * sinPitch + tmpZ * cosPitch;
    }

    // Rz
    {
      double tmpX = newX, tmpY = newY;
      newX = tmpX * cosYaw - tmpY * sinYaw;
      newY = tmpX * sinYaw + tmpY * cosYaw;
    }

    return Coord3D(newX, newY, newZ) + rotationPoint;
  }

};

/**
 * Operator definition for convenient printing of a Coord3D object to a Stream.
 */
static std::ostream &operator<<(std::ostream &stream, const Coord3D &obj) {
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
