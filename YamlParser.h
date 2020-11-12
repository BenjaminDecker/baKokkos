//
// Created by Benjamin Decker on 02/11/2020.
//

#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>
#include <optional>
#include "Coord3D.h"
#include "ParticleGroup.h"

/**
 * @brief and stores ParticleGroups from a given .yaml file
 * @see ParticleGroup
 */
class YamlParser {
 public:
  std::optional<int> iterations;
  std::optional<double> deltaT;
  std::optional<double> cutoff;
  std::optional<std::pair<Coord3D, Coord3D>> box;
  std::optional<std::string> vtkFileName;
  std::optional<int> vtkWriteFrequency;
  std::optional<std::string> yamlFileName;
  std::vector<ParticleCuboid> particleCuboids; /**< Cuboids that could be read from the .yamlFile */
  std::vector<ParticleSphere> particleSpheres; /**< Spheres that could be read from the .yamlFile */

  explicit YamlParser(const std::string &fileName) {
    YAML::Node config = YAML::LoadFile(fileName);

    if(config["cutoff"]) {
      cutoff = config["cutoff"].as<double>();
    }
    if(config["box-min"] && config["box-max"]) {
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
    }
  }
};



