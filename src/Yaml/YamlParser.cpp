//
// Created by Benjamin Decker on 02/11/2020.
//

#include "YamlParser.h"
YamlParser::YamlParser(const std::string &fileName) {
  YAML::Node config = YAML::LoadFile(fileName);

  if (config["iterations"]) {
    iterations = config["iterations"].as<int>();
  }
  if (config["deltaT"]) {
    deltaT = config["deltaT"].as<double>();
  }
  if (config["cutoff"]) {
    cutoff = config["cutoff"].as<double>();
  }
  if (config["vtk-filename"]) {
    vtkFileName = config["vtk-filename"].as<std::string>();
  }
  if (config["vtk-write-frequency"]) {
    vtkWriteFrequency = config["vtk-write-frequency"].as<int>();
  }
  if (config["box-min"] && config["box-max"]) {
    Coord3D boxMin = Coord3D(config["box-min"][0].as<double>(),
                             config["box-min"][1].as<double>(),
                             config["box-min"][2].as<double>());
    Coord3D boxMax = Coord3D(config["box-max"][0].as<double>(),
                             config["box-max"][1].as<double>(),
                             config["box-max"][2].as<double>());
    box = {boxMin, boxMax};
  }

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

    if (objectIterator->first.as<std::string>() == "CubeClosestPacked") {
      for (auto cubeClosestIterator = objectIterator->second.begin();
           cubeClosestIterator != objectIterator->second.end();
           ++cubeClosestIterator) {
        auto cubeClosest = cubeClosestIterator->second;
        auto typeID = cubeClosest["particle-type"].as<int>();
        auto spacing = cubeClosest["particle-spacing"].as<double>();
        auto particleEpsilon = cubeClosest["particle-epsilon"].as<double>();
        auto particleSigma = cubeClosest["particle-sigma"].as<double>();
        auto particleMass = cubeClosest["particle-mass"].as<double>();

        auto velocityNode = cubeClosest["velocity"];
        auto velocity = Coord3D(velocityNode[0].as<double>(),
                                velocityNode[1].as<double>(),
                                velocityNode[2].as<double>());
        auto bottomLeftCornerNode = cubeClosest["bottomLeftCorner"];
        auto bottomLeftCorner = Coord3D(bottomLeftCornerNode[0].as<double>(),
                                        bottomLeftCornerNode[1].as<double>(),
                                        bottomLeftCornerNode[2].as<double>());
        auto boxLengthNode = cubeClosest["box-length"];
        auto boxLength = Coord3D(boxLengthNode[0].as<double>(),
                                 boxLengthNode[1].as<double>(),
                                 boxLengthNode[2].as<double>());

        cubesClosest.emplace_back(typeID,
                                     spacing,
                                     velocity,
                                     particleEpsilon,
                                     particleSigma,
                                     particleMass,
                                     bottomLeftCorner,
                                     boxLength);
      }
    }
  }
}

std::vector<Particle> YamlParser::getParticles() const {
  std::vector<Particle> particles;
  for (auto &cuboid : particleCuboids) {
    for (auto &newParticle : cuboid.getParticles(particles.size())) {
      particles.push_back(newParticle);
    }
  }
  for (auto &sphere : particleSpheres) {
    for (auto &newParticle : sphere.getParticles(particles.size())) {
      particles.push_back(newParticle);
    }
  }
  for (auto &cubeClosest : cubesClosest) {
    for (auto &newParticle : cubeClosest.getParticles(particles.size())) {
      particles.push_back(newParticle);
    }
  }
  return particles;
}
