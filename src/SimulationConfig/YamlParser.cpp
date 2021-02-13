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
  static constexpr auto gaussianGeneratorStr = "GaussianGenerator";
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
  static constexpr auto boxLengthStr = "box-length";
  static constexpr auto numParticlesStr = "num-particles";
  static constexpr auto distributionMeanStr = "distribution-mean";
  static constexpr auto distributionStdDevStr = "distribution-std-dev";
  YAML::Node config = YAML::LoadFile(fileName);
  if (config[iterationsStr]) {
    iterations = config[iterationsStr].as<int>();
  }
  if (config[deltaTStr]) {
    deltaT = config[deltaTStr].as<float>();
  }
  if (config[cutoffStr]) {
    cutoff = config[cutoffStr].as<float>();
  }
  if (config[vtk_filenameStr]) {
    vtkFileName = config[vtk_filenameStr].as<std::string>();
  }
  if (config[vtk_write_frequencyStr]) {
    vtkWriteFrequency = config[vtk_write_frequencyStr].as<int>();
  }
  if (config[globalForceStr]) {
    globalForce = Coord3D(config[globalForceStr][0].as<float>(),
                          config[globalForceStr][1].as<float>(),
                          config[globalForceStr][2].as<float>());
  }
  if (config[box_minStr] && config[box_maxStr]) {
    Coord3D boxMin = Coord3D(config[box_minStr][0].as<float>(),
                             config[box_minStr][1].as<float>(),
                             config[box_minStr][2].as<float>());
    Coord3D boxMax = Coord3D(config[box_maxStr][0].as<float>(),
                             config[box_maxStr][1].as<float>(),
                             config[box_maxStr][2].as<float>());
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
        auto spacing = cuboid[particleSpacingStr].as<float>();
        auto particleEpsilon = cuboid[particleEpsilonStr].as<float>();
        auto particleSigma = cuboid[particleSigmaStr].as<float>();
        auto particleMass = cuboid[particleMassStr].as<float>();

        auto velocityNode = cuboid[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<float>(),
                                velocityNode[1].as<float>(),
                                velocityNode[2].as<float>());
        auto bottomLeftCornerNode = cuboid[bottomLeftCornerStr];
        auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<float>(),
                                        bottomLeftCornerNode[1].as<float>(),
                                        bottomLeftCornerNode[2].as<float>());
        auto particlesPerDimensionNode = cuboid[particles_per_dimensionStr];
        auto particlesPerDimension = Coord3D(particlesPerDimensionNode[0].as<float>(),
                                             particlesPerDimensionNode[1].as<float>(),
                                             particlesPerDimensionNode[2].as<float>());

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
        auto spacing = sphere[particleSpacingStr].as<float>();
        auto particleEpsilon = sphere[particleEpsilonStr].as<float>();
        auto particleSigma = sphere[particleSigmaStr].as<float>();
        auto particleMass = sphere[particleMassStr].as<float>();
        auto radius = sphere[radiusStr].as<float>();

        auto velocityNode = sphere[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<float>(),
                                velocityNode[1].as<float>(),
                                velocityNode[2].as<float>());
        auto centerNode = sphere[centerStr];
        auto center = Coord3D(centerNode[0].as<float>(),
                              centerNode[1].as<float>(),
                              centerNode[2].as<float>());

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
        auto spacing = cubeClosest[particleSpacingStr].as<float>();
        auto particleEpsilon = cubeClosest[particleEpsilonStr].as<float>();
        auto particleSigma = cubeClosest[particleSigmaStr].as<float>();
        auto particleMass = cubeClosest[particleMassStr].as<float>();

        auto velocityNode = cubeClosest[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<float>(),
                                velocityNode[1].as<float>(),
                                velocityNode[2].as<float>());
        auto bottomLeftCornerNode = cubeClosest[bottomLeftCornerStr];
        auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<float>(),
                                        bottomLeftCornerNode[1].as<float>(),
                                        bottomLeftCornerNode[2].as<float>());
        auto boxLengthNode = cubeClosest[boxLengthStr];
        auto boxLength = Coord3D(boxLengthNode[0].as<float>(),
                                 boxLengthNode[1].as<float>(),
                                 boxLengthNode[2].as<float>());

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

    if (objectIterator->first.as<std::string>() == gaussianGeneratorStr) {
      for (auto gaussianGeneratorIterator = objectIterator->second.begin();
           gaussianGeneratorIterator != objectIterator->second.end();
           ++gaussianGeneratorIterator) {
        auto gaussianGenerator = gaussianGeneratorIterator->second;
        auto typeID = gaussianGenerator[particleTypeStr].as<int>();
        auto numParticles = gaussianGenerator[numParticlesStr].as<int>();
        auto boxMinNode = gaussianGenerator[box_minStr];
        auto boxMin = Coord3D(boxMinNode[0].as<float>(),
                              boxMinNode[1].as<float>(),
                              boxMinNode[2].as<float>());
        auto boxMaxNode = gaussianGenerator[box_maxStr];
        auto boxMax = Coord3D(boxMaxNode[0].as<float>(),
                              boxMaxNode[1].as<float>(),
                              boxMaxNode[2].as<float>());
        auto distributionMeanNode = gaussianGenerator[distributionMeanStr];
        auto distributionMean = Coord3D(distributionMeanNode[0].as<float>(),
                                        distributionMeanNode[1].as<float>(),
                                        distributionMeanNode[2].as<float>());
        auto distributionStdDevNode = gaussianGenerator[distributionStdDevStr];
        auto distributionStdDev = Coord3D(distributionStdDevNode[0].as<float>(),
                                          distributionStdDevNode[1].as<float>(),
                                          distributionStdDevNode[2].as<float>());
        stdDev = distributionStdDev.x;
        auto particleEpsilon = gaussianGenerator[particleEpsilonStr].as<float>();
        auto particleSigma = gaussianGenerator[particleSigmaStr].as<float>();
        auto particleMass = gaussianGenerator[particleMassStr].as<float>();

        auto velocityNode = gaussianGenerator[velocityStr];
        auto velocity = Coord3D(velocityNode[0].as<float>(),
                                velocityNode[1].as<float>(),
                                velocityNode[2].as<float>());

        particleGroups.push_back(
            std::make_shared<GaussianGenerator>(
                typeID,
                numParticles,
                velocity,
                particleEpsilon,
                particleSigma,
                particleMass,
                boxMin,
                boxMax,
                distributionMean,
                distributionStdDev));
      }
    }
  }
}
