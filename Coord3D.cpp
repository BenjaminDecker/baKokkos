//
// Created by Benjamin Decker on 22.10.20.
//

#include "Coord3D.h"

KOKKOS_FUNCTION
Coord3D::Coord3D() : x(0), y(0), z(0) {}

KOKKOS_FUNCTION
Coord3D::Coord3D(const Coord3D &rhs) {
  x = rhs.x;
  y = rhs.y;
  z = rhs.z;
}

KOKKOS_FUNCTION
Coord3D::Coord3D(double x, double y, double z) : x(x), y(y), z(z) {}

KOKKOS_FUNCTION
Coord3D::~Coord3D() = default;

KOKKOS_FUNCTION
Coord3D Coord3D::distanceTo(const Coord3D &other) const {
  return Coord3D{x - other.x, y - other.y, z - other.z};
}

KOKKOS_FUNCTION
double Coord3D::absoluteValue() const {
  return std::sqrt(x * x + y * y + z * z);
}

KOKKOS_FUNCTION
Coord3D Coord3D::operator+(const Coord3D &rhs) const {
  return Coord3D{x + rhs.x, y + rhs.y, z + rhs.z};
}

KOKKOS_FUNCTION
Coord3D Coord3D::operator*(const double rhs) const {
  return Coord3D{x * rhs, y * rhs, z * rhs};
}

KOKKOS_FUNCTION
Coord3D Coord3D::operator/(const double &rhs) const {
  return Coord3D{x / rhs, y / rhs, z / rhs};
}

KOKKOS_FUNCTION
bool Coord3D::operator==(const Coord3D &rhs) const {
  return x == rhs.x && y == rhs.y && z == rhs.z;
}

KOKKOS_FUNCTION
Coord3D &Coord3D::operator+=(const Coord3D &rhs) {
  x = x + rhs.x;
  y = y + rhs.y;
  z = z + rhs.z;
  return *this;
}

KOKKOS_FUNCTION
void Coord3D::operator+=(volatile const Coord3D &rhs) volatile {
  x = x + rhs.x;
  y = y + rhs.y;
  z = z + rhs.z;
}

KOKKOS_FUNCTION
std::ostream &operator<<(std::ostream &stream, const Coord3D &obj) {
  stream << "( " << obj.x << ", " << obj.y << ", " << obj.z << " )";
  return stream;
}
