//
// Created by ffbde on 16/10/2020.
//

#ifndef BAKOKKOS_PARTICLE_H
#define BAKOKKOS_PARTICLE_H

struct Coord3D {
    double x, y, z;
    Coord3D operator + (const Coord3D& other) const {
        return Coord3D {x + other.x, y + other.y, z + other.z};
    }
    Coord3D operator * (const double& other) const {
        return Coord3D {x * other, y * other, z * other};
    }
};

class Particle {
public:
    Particle(double x, double y, double z);
    Particle(Coord3D position);

    Coord3D position = {0.0, 0.0, 0.0};
    Coord3D velocity = {0.0, 0.0, 0.0};
};


#endif //BAKOKKOS_PARTICLE_H
