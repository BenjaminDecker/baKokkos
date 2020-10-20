//
// Created by ffbde on 16/10/2020.
//

#pragma once

#include <vector>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>

struct Coord3D {
  double x, y, z;

  Coord3D(double x, double y, double z) : x(x), y(y), z(z) {}

  Coord3D() : x(0), y(0), z(0) {}

  Coord3D distanceTo(const Coord3D &other) const {
    return Coord3D{x - other.x, y - other.y, z - other.z};
  }
  double absoluteValue() const {
    return std::sqrt(x * x + y * y + z * z);
  }
  Coord3D operator+(const Coord3D &rhs) const {
    return Coord3D{x + rhs.x, y + rhs.y, z + rhs.z};
  }
  Coord3D operator*(const double rhs) const {
    return Coord3D{x * rhs, y * rhs, z * rhs};
  }
  Coord3D operator/(const double &rhs) const {
    return Coord3D{x / rhs, y / rhs, z / rhs};
  }
  bool operator==(const Coord3D &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
  Coord3D &operator+=(const Coord3D &rhs) {
    x = x + rhs.x;
    y = y + rhs.y;
    z = z + rhs.z;
    return *this;
  }
  volatile Coord3D &operator+=(volatile const Coord3D &rhs) volatile {
    x = x + rhs.x;
    y = y + rhs.y;
    z = z + rhs.z;
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &stream, const Coord3D &obj) {
    stream << "( " << obj.x << ", " << obj.y << ", " << obj.z << " )";
    return stream;
  }
};

typedef Kokkos::View<Coord3D *> Coord3DView;

class ParticleContainer {
 public:
  int size;
  Coord3DView positions;
  Coord3DView forces;
  Coord3DView velocities;

  explicit ParticleContainer(int size);
};
