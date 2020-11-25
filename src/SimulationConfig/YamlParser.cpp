//
// Created by Benjamin Decker on 02/11/2020.
//

#include "YamlParser.h"
YamlParser::YamlParser(const std::string &fileName) {
  static constexpr auto iterationsStr = "iterations";
  static constexpr auto deltaTStr = "deltaT";
  static constexpr auto cutoffStr = "cutoff";
  static constexpr auto vtk_filenameStr = "vtk-filename";
  static constexpr auto vtk_write_frequencyStr = "vtk-write-frequency";
  static constexpr auto globalForceStr = "globalForce";
  static constexpr auto box_minStr = "box-min";
  static constexpr auto box_maxStr = "box-max";
  static constexpr auto objectsStr = "Objects";
  static constexpr auto cubeGridStr = "CubeGrid";
  static constexpr auto sphereStr = "Sphere";
  static constexpr auto cubeClosestPackedStr = "CubeClosestPacked";
  static constexpr auto particleTypeStr = "particle-type";
  static constexpr auto particleSpacingStr = "particle-spacing";
  static constexpr auto particleEpsilonStr = "particle-epsilon";
  static constexpr auto particleSigmaStr = "particle-sigma";
  static constexpr auto particleMassStr = "particle-mass";
  static constexpr auto velocityStr = "velocity";
  static constexpr auto bottomLeftCornerStr = "bottomLeftCorner";
  static constexpr auto particles_per_dimensionStr = "particles-per-dimension";
  static constexpr auto radiusStr = "radius";
  static constexpr auto centerStr = "center";
  static constexpr auto boxLengthStr = "boxLength";
  YAML::Node config = YAML::LoadFile(fileName);
  if (config[iterationsStr]) {
    iterations = config[iterationsStr].as<int>();
  }
  if (config[deltaTStr]) {
    deltaT = config[deltaTStr].as<double>();
  }
  if (config[cutoffStr]) {
    cutoff = config[cutoffStr].as<double>();
  }
  if (config[vtk_filenameStr]) {
    vtkFileName = config[vtk_filenameStr].as<std::string>();
  }
  if (config[vtk_write_frequencyStr]) {
    vtkWriteFrequency = config[vtk_write_frequencyStr].as<int>();
  }
  if (config[globalForceStr]) {
    globalForce = Coord3D(config[globalForceStr][0].as<double>(),
                          config[globalForceStr][1].as<double>(),
                          config[globalForceStr][2].as<double>());
  }
  if (config[box_minStr] && config[box_maxStr]) {
    Coord3D boxMin = Coord3D(config[box_minStr][0].as<double>(),
                             config[box_minStr][1].as<double>(),
                             config[box_minStr][2].as<double>());
    Coord3D boxMax = Coord3D(config[box_maxStr][0].as<double>(),
                             config[box_maxStr][1].as<double>(),
                             config[box_maxStr][2].as<double>());
    box = {boxMin, boxMax};
  }

  auto objects = config[objectsStr];
  for (auto objectIterator = objects.begin();
       objectIterator != objects.end();
       ++objectIterator) {
    if (objectIterator->first.as<std::string>() == cubeGridStr) {
      for (auto cuboidIterator = objectIterator->second.begin();
           cuboidIterator != objectIterator->second.end();
           ++cuboidIterator) {
        auto cuboid = cuboidIterator->second;
        auto typeID = cuboid[particleTypeStr].as<int>();
        auto spacing = cuboid[particleSpacingStr].as<double>();
        auto particleEpsilon = cuboid[particleEpsilonStr].as<double>();
        auto particleSigma = cuboid[particleSigmaStr].as<double>();
        auto particleMass = cuboid[particleMassStr].as<double>();

        auto velocityNode = cuboid[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<double>(),
                                velocityNode[1].as<double>(),
                                velocityNode[2].as<double>());
        auto bottomLeftCornerNode = cuboid[bottomLeftCornerStr];
        auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<double>(),
                                        bottomLeftCornerNode[1].as<double>(),
                                        bottomLeftCornerNode[2].as<double>());
        auto particlesPerDimensionNode = cuboid[particles_per_dimensionStr];
        auto particlesPerDimension = Coord3D(particlesPerDimensionNode[0].as<double>(),
                                             particlesPerDimensionNode[1].as<double>(),
                                             particlesPerDimensionNode[2].as<double>());

        particleGroups.push_back(
            std::make_shared<ParticleCuboid>(
                typeID,
                spacing,
                velocity,
                particleEpsilon,
                particleSigma,
                particleMass,
                bottomLeftCorner,
                particlesPerDimension));
      }
    }

    if (objectIterator->first.as<std::string>() == sphereStr) {
      for (auto sphereIterator = objectIterator->second.begin();
           sphereIterator != objectIterator->second.end();
           ++sphereIterator) {
        auto sphere = sphereIterator->second;
        auto typeID = sphere[particleTypeStr].as<int>();
        auto spacing = sphere[particleSpacingStr].as<double>();
        auto particleEpsilon = sphere[particleEpsilonStr].as<double>();
        auto particleSigma = sphere[particleSigmaStr].as<double>();
        auto particleMass = sphere[particleMassStr].as<double>();
        auto radius = sphere[radiusStr].as<double>();

        auto velocityNode = sphere[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<double>(),
                                velocityNode[1].as<double>(),
                                velocityNode[2].as<double>());
        auto centerNode = sphere[centerStr];
        auto center = Coord3D(centerNode[0].as<double>(),
                              centerNode[1].as<double>(),
                              centerNode[2].as<double>());

        particleGroups.push_back(
            std::make_shared<ParticleSphere>(
                typeID,
                spacing,
                velocity,
                particleEpsilon,
                particleSigma,
                particleMass,
                center,
                radius));
      }
    }

    if (objectIterator->first.as<std::string>() == cubeClosestPackedStr) {
      for (auto cubeClosestIterator = objectIterator->second.begin();
           cubeClosestIterator != objectIterator->second.end();
           ++cubeClosestIterator) {
        auto cubeClosest = cubeClosestIterator->second;
        auto typeID = cubeClosest[particleTypeStr].as<int>();
        auto spacing = cubeClosest[sphereStr].as<double>();
        auto particleEpsilon = cubeClosest[particleEpsilonStr].as<double>();
        auto particleSigma = cubeClosest[particleSigmaStr].as<double>();
        auto particleMass = cubeClosest[particleMassStr].as<double>();

        auto velocityNode = cubeClosest[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<double>(),
                                velocityNode[1].as<double>(),
                                velocityNode[2].as<double>());
        auto bottomLeftCornerNode = cubeClosest[bottomLeftCornerStr];
        auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<double>(),
                                        bottomLeftCornerNode[1].as<double>(),
                                        bottomLeftCornerNode[2].as<double>());
        auto boxLengthNode = cubeClosest[boxLengthStr];
        auto boxLength = Coord3D(boxLengthNode[0].as<double>(),
                                 boxLengthNode[1].as<double>(),
                                 boxLengthNode[2].as<double>());

        particleGroups.push_back(
            std::make_shared<CubeClosestPacked>(
                typeID,
                spacing,
                velocity,
                particleEpsilon,
                particleSigma,
                particleMass,
                bottomLeftCorner,
                boxLength));
      }
    }
  }
}
