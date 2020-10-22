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

  KOKKOS_FUNCTION
  Coord3D();

  KOKKOS_FUNCTION
  Coord3D(const Coord3D &rhs);

  KOKKOS_FUNCTION
  Coord3D(double x, double y, double z);

  KOKKOS_FUNCTION
  ~Coord3D();

  KOKKOS_FUNCTION
  Coord3D distanceTo(const Coord3D &other) const;

  KOKKOS_FUNCTION
  double absoluteValue() const;

  KOKKOS_FUNCTION
  Coord3D operator+(const Coord3D &rhs) const;

  KOKKOS_FUNCTION
  Coord3D operator*(const double rhs) const;

  KOKKOS_FUNCTION
  Coord3D operator/(const double &rhs) const;

  KOKKOS_FUNCTION
  bool operator==(const Coord3D &rhs) const;

  KOKKOS_FUNCTION
  Coord3D &operator+=(const Coord3D &rhs);

  KOKKOS_FUNCTION
  void operator+=(const volatile Coord3D &rhs) volatile;

  KOKKOS_FUNCTION
  friend std::ostream &operator<<(std::ostream &stream, const Coord3D &obj);
};