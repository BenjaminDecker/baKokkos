//
// Created by Benjamin Decker on 02/11/2020.
//

#include "YamlParser.h"
YamlParser::YamlParser(const std::string &fileName) {
  YAML::Node config = YAML::LoadFile(fileName);

  auto objects = config["Objects"];
  for (auto objectIterator = objects.begin();
       objectIterator != objects.end();
       ++objectIterator) {
    if (objectIterator->first.as<std::string>() == "CubeGrid") {
      for (auto cuboidIterator = objectIterator->second.begin();
           cuboidIterator != objectIterator->second.end();
           ++cuboidIterator) {
        auto cuboid = cuboidIterator->second;
        auto typeID = cuboid["particle-type"].as<int>();
        auto spacing = cuboid["particle-spacing"].as<double>();
        auto particleEpsilon = cuboid["particle-epsilon"].as<double>();
        auto particleSigma = cuboid["particle-sigma"].as<double>();
        auto particleMass = cuboid["particle-mass"].as<double>();

        auto velocityNode = cuboid["velocity"];
        auto velocity = Coord3D(velocityNode[0].as<double>(),
                                velocityNode[1].as<double>(),
                                velocityNode[2].as<double>());
        auto bottomLeftCornerNode = cuboid["bottomLeftCorner"];
        auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<double>(),
                                        bottomLeftCornerNode[1].as<double>(),
                                        bottomLeftCornerNode[2].as<double>());
        auto particlesPerDimensionNode = cuboid["particles-per-dimension"];
        auto particlesPerDimension = Coord3D(particlesPerDimensionNode[0].as<double>(),
                                             particlesPerDimensionNode[1].as<double>(),
                                             particlesPerDimensionNode[2].as<double>());

        particleCuboids.emplace_back(typeID,
                                     spacing,
                                     velocity,
                                     particleEpsilon,
                                     particleSigma,
                                     particleMass,
                                     bottomLeftCorner,
                                     particlesPerDimension);
      }
    }

    if (objectIterator->first.as<std::string>() == "Sphere") {
      for (auto sphereIterator = objectIterator->second.begin();
           sphereIterator != objectIterator->second.end();
           ++sphereIterator) {
        auto sphere = sphereIterator->second;
        auto typeID = sphere["particle-type"].as<int>();
        auto spacing = sphere["particle-spacing"].as<double>();
        auto particleEpsilon = sphere["particle-epsilon"].as<double>();
        auto particleSigma = sphere["particle-sigma"].as<double>();
        auto particleMass = sphere["particle-mass"].as<double>();
        auto radius = sphere["radius"].as<double>();

        auto velocityNode = sphere["velocity"];
        auto velocity = Coord3D(velocityNode[0].as<double>(),
                                velocityNode[1].as<double>(),
                                velocityNode[2].as<double>());
        auto centerNode = sphere["center"];
        auto center = Coord3D(centerNode[0].as<double>(),
                              centerNode[1].as<double>(),
                              centerNode[2].as<double>());

        particleSpheres.emplace_back(typeID,
                                     spacing,
                                     velocity,
                                     particleEpsilon,
                                     particleSigma,
                                     particleMass,
                                     center,
                                     radius);
      }
    }
  }

//  auto cuboids = objects["CubeGrid"];
//  for (int i = 0; i < cuboids.size(); ++i) {
//    auto cuboid = cuboids[i];
//    auto typeID = cuboid["particle-type"].as<int>();
//    auto spacing = cuboid["particle-spacing"].as<double>();
//    auto particleEpsilon = cuboid["particle-epsilon"].as<double>();
//    auto particleSigma = cuboid["particle-sigma"].as<double>();
//    auto particleMass = cuboid["particle-mass"].as<double>();
//
//    auto velocityNode = cuboid["velocity"];
//    auto velocity = Coord3D(velocityNode[0].as<double>(),
//                            velocityNode[1].as<double>(),
//                            velocityNode[2].as<double>());
//    auto bottomLeftCornerNode = cuboid["bottomLeftCorner"];
//    auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<double>(),
//                                    bottomLeftCornerNode[1].as<double>(),
//                                    bottomLeftCornerNode[2].as<double>());
//    auto particlesPerDimensionNode = cuboid["particles-per-dimension"];
//    auto particlesPerDimension = Coord3D(particlesPerDimensionNode[0].as<double>(),
//                                         particlesPerDimensionNode[1].as<double>(),
//                                         particlesPerDimensionNode[2].as<double>());
//
//    particleCuboids.emplace_back(typeID,
//                                 spacing,
//                                 velocity,
//                                 particleEpsilon,
//                                 particleSigma,
//                                 particleMass,
//                                 bottomLeftCorner,
//                                 particlesPerDimension);
//  }
//
//  auto spheres = objects["Sphere"];
//  for (int i = 0; i < spheres.size(); ++i) {
//    auto sphere = spheres[i];
//    auto typeID = sphere["particle-type"].as<int>();
//    auto spacing = sphere["particle-spacing"].as<double>();
//    auto particleEpsilon = sphere["particle-epsilon"].as<double>();
//    auto particleSigma = sphere["particle-sigma"].as<double>();
//    auto particleMass = sphere["particle-mass"].as<double>();
//    auto radius = sphere["radius"].as<double>();
//
//    auto velocityNode = sphere["velocity"];
//    auto velocity = Coord3D(velocityNode[0].as<double>(),
//                            velocityNode[1].as<double>(),
//                            velocityNode[2].as<double>());
//    auto centerNode = sphere["center"];
//    auto center = Coord3D(centerNode[0].as<double>(),
//                          centerNode[1].as<double>(),
//                          centerNode[2].as<double>());
//
//    particleSpheres.emplace_back(typeID,
//                                 spacing,
//                                 velocity,
//                                 particleEpsilon,
//                                 particleSigma,
//                                 particleMass,
//                                 center,
//                                 radius);
//  }
}
