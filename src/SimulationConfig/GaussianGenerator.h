//
// Created by Benjamin Decker on 11.02.21.
//

#pragma once

#include <random>

/**
 * Generator class for gaussian distributions
 */
class GaussianGenerator : public ParticleGroup {
 public:

  const int numParticles;
  const Coord3D boxMin;
  const Coord3D boxMax;
  const Coord3D distributionMean;
  const Coord3D distributionStdDev;

  GaussianGenerator(int typeID,
                    int numParticles,
                    Coord3D velocity,
                    float particleEpsilon,
                    float particleSigma,
                    float particleMass,
                    Coord3D boxMin,
                    Coord3D boxMax,
                    Coord3D distributionMean,
                    Coord3D distributionStdDev)
      : ParticleGroup(typeID, -1, velocity, particleEpsilon, particleSigma, particleMass),
        numParticles(numParticles),
        boxMin(boxMin),
        boxMax(boxMax),
        distributionMean(distributionMean),
        distributionStdDev(distributionStdDev) {}

  [[nodiscard]] std::vector<Particle> getParticles(int startID = 0) const override {
    std::vector<Particle> particles;
    std::default_random_engine generator(42);
    std::array<std::normal_distribution<float>, 3> distributions = {
        std::normal_distribution<float>{distributionMean.x, distributionStdDev.x},
        std::normal_distribution<float>{distributionMean.y, distributionStdDev.y},
        std::normal_distribution<float>{distributionMean.z, distributionStdDev.z}};

    for (unsigned long i = 0; i < numParticles; ++i) {
      Coord3D position (distributions[0](generator),
                        distributions[1](generator),
                        distributions[2](generator));
      // verifies that position is valid
      for (size_t attempts = 1; attempts <= _maxAttempts and (not isInBox(position));
           ++attempts) {
        if (attempts == _maxAttempts) {
          std::ostringstream errormessage;
          errormessage << "GaussianGenerator::fillWithParticles(): Could not find a valid position for particle " << i
                       << " after " << _maxAttempts << " attempts. Check if your parameters make sense:" << std::endl
                       << "BoxMin       = " << boxMin << std::endl
                       << "BoxMax       = " << boxMax << std::endl
                       << "Gauss mean   = " << distributionMean << std::endl
                       << "Gauss stdDev = " << distributionStdDev << std::endl;
          throw std::runtime_error(errormessage.str());
        }
        position = Coord3D(distributions[0](generator),
                           distributions[1](generator),
                           distributions[2](generator));
      };
      particles.emplace_back(i + startID, typeID, position, velocity);
    }
    return particles;
  }

  [[nodiscard]] std::vector<Particle> getParticles() const override {
    return getParticles(0);
  }

 private:
  /**
   * Maximum number of attempts the random generator gets to find a valid position before considering the input to be
   * bad
   */
  constexpr static size_t _maxAttempts = 100;
  [[nodiscard]] bool isInBox(const Coord3D &position) const {
    return boxMin.x < position.x &&
        boxMin.y < position.y &&
        boxMin.z < position.z &&
        position.x < boxMax.x &&
        position.y < boxMax.y &&
        position.z < boxMax.z;
  }
};
