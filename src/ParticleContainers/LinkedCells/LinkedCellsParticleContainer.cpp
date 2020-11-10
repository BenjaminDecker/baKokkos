//
// Created by Benjamin Decker on 08.11.20.
//

#include <map>
#include "LinkedCellsParticleContainer.h"

LinkedCellsParticleContainer::LinkedCellsParticleContainer(const YamlParser &parser)
    : cutoff(parser.cutoff), boxMin(parser.boxMin), boxMax(parser.boxMax) {
  std::vector<std::vector<Particle>> cuboids;
  std::vector<std::vector<Particle>> spheres;

  for (auto &cuboid : parser.particleCuboids) {
    cuboids.emplace_back();
    cuboid.getParticles(cuboids.at(cuboids.size() - 1));
  }
  for (auto &sphere : parser.particleSpheres) {
    spheres.emplace_back();
    sphere.getParticles(spheres.at(spheres.size() - 1));
  }
  size = 0;
  for (auto &cuboid : cuboids) {
    size += cuboid.size();
  }
  for (auto &sphere : spheres) {
    size += sphere.size();
  }

  Coord3D boxSize = boxMax - boxMin;
  numCellsX = static_cast<int>(boxSize.x / cutoff);
  numCellsY = static_cast<int>(boxSize.y / cutoff);
  numCellsZ = static_cast<int>(boxSize.z / cutoff);
  int numCells = numCellsX * numCellsY * numCellsZ;

}

Particle LinkedCellsParticleContainer::getParticle(int id) const {
  //TODO
  return Particle(0);
}

void LinkedCellsParticleContainer::insertParticle(const Particle &particle, int id) const {
  //TODO
}

int LinkedCellsParticleContainer::getIndexOf(int x, int y, int z) const {
  return z * numCellsX * numCellsY + y * numCellsX + x;
}

int LinkedCellsParticleContainer::calculateCell(const Coord3D &position) const {
  if(position.x < boxMin.x || position.y < boxMin.y || position.z < boxMin.z) {
    return -1;
  }
  int cellX = static_cast<int>((position.x - boxMin.x) / cutoff);
  int cellY = static_cast<int>((position.y - boxMin.y) / cutoff);
  int cellZ = static_cast<int>((position.z - boxMin.z) / cutoff);
  return getIndexOf(cellX, cellY, cellZ);
}
