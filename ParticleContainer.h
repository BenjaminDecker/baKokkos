//
// Created by ffbde on 16/10/2020.
//

#pragma once

#include "Coord3D.h"

typedef Kokkos::View<Coord3D *> Coord3DView;

class ParticleContainer {
 public:
  int size;
  Coord3DView positions;
  Coord3DView forces;
  Coord3DView velocities;

  explicit ParticleContainer(int size);
};
