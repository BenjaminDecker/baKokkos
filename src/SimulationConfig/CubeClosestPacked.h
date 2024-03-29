//
// Created by Benjamin Decker on 15.11.20.
//

#pragma once

#include "ParticleGroup.h"

class CubeClosestPacked : public ParticleGroup {
 public:

  /**
   * Position of the bottom left corner of the cuboid
   */
  const Coord3D bottomLeftCorner;

  /**
   * Side lengths of the cuboid
   */
  const Coord3D boxLength;

  CubeClosestPacked(int typeID,
                    float spacing,
                    Coord3D velocity,
                    float particleEpsilon,
                    float particleSigma,
                    float particleMass,
                    Coord3D bottomLeftCorner,
                    Coord3D boxLength)
      : ParticleGroup(typeID, spacing, velocity, particleEpsilon, particleSigma, particleMass),
        bottomLeftCorner(bottomLeftCorner),
        boxLength(boxLength) {}

  [[nodiscard]] std::vector<Particle> getParticles() const override {
    return getParticles(0);
  }

  [[nodiscard]] std::vector<Particle> getParticles(int startID) const override {
    // Spacing in y direction when only moving 60° on the unit circle. Or the height in an equilateral triangle.
    const float spacingRow = spacing * std::sqrt(3. / 4.);
    // Spacing in z direction. Height in an equilateral tetraeder.
    const float spacingLayer = spacing * std::sqrt(2. / 3.);
    // Shorter part of the bisectrix when split at the intersection of all bisectrices.
    const float xOffset = spacing * 1. / 2.;
    // Shorter part of the bisectrix when split at the intersection of all bisectrices.
    const float yOffset = spacing * std::sqrt(1. / 12.);

    // The packing alternates between odd and even layers and rows
    bool evenLayer = true;
    bool evenRow = true;

    std::vector<Particle> particles;
    size_t id = startID;
    for (float z = bottomLeftCorner.z; z < bottomLeftCorner.z + boxLength.z; z += spacingLayer) {
      float starty = evenLayer ? bottomLeftCorner.y : bottomLeftCorner.y + yOffset;
      for (float y = starty; y < bottomLeftCorner.y + boxLength.y; y += spacingRow) {
        float startx = evenRow ? bottomLeftCorner.x : bottomLeftCorner.x + xOffset;
        for (float x = startx; x < bottomLeftCorner.x + boxLength.x; x += spacing) {
          particles.emplace_back(id++, typeID, Coord3D(x, y, z), velocity);
        }
        evenRow = not evenRow;
      }
      evenLayer = not evenLayer;
    }
    return particles;
  }
};
