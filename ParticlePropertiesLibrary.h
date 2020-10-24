//
// Created by Benjamin Decker on 18/10/2020.
//

#pragma once

#include <vector>

struct ParticleType {
  int typeID;
  double epsilon;
  double sigma;
  double mass;
};

class ParticlePropertiesLibrary {
  std::vector<ParticleType> particleProperties;
 public:
  void addParticleType(int typeID, double epsilon, double sigma, double mass) {
    particleProperties.push_back(ParticleType{typeID, epsilon, sigma, mass});
  }

  double getEpsilon(int typeID) {
    return particleProperties.at(typeID).epsilon;
  }
  double getSigma(int typeID) {
    return particleProperties.at(typeID).sigma;
  }
  double getMass(int typeID) {
    return particleProperties.at(typeID).mass;
  }
};



