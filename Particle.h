//
// Created by ffbde on 16/10/2020.
//

#pragma once

#include <cmath>


struct Coord3D {
    double x, y, z;

    Coord3D distanceTo(const Coord3D& other) const {
        return Coord3D{ x - other.x, y - other.y, z - other.z };
    }
    double absoluteValue() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Coord3D operator + (const Coord3D& rhs) const {
        return Coord3D {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    Coord3D operator * (const double& rhs) const {
        return Coord3D {x * rhs, y * rhs, z * rhs};
    }
    Coord3D operator / (const double& rhs) const {
        return Coord3D {x / rhs, y / rhs, z / rhs};
    }
    bool operator == (const Coord3D& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
    Coord3D& operator += (const Coord3D& rhs) {
        x = x + rhs.x;
        y = y + rhs.y;
        z = z + rhs.z;
        return *this;
    }
};

class Particle {
public:
    int id;
    int typeID;
    Coord3D position = {0.0, 0.0, 0.0};
    Coord3D force = {0.0, 0.0, 0.0};
    Coord3D velocity = {0.0, 0.0, 0.0};

    explicit Particle(int id, int typeID, Coord3D position) : id(id), typeID(typeID), position(position){};
    Particle(int id, int typeID, Coord3D position, Coord3D velocity) : id(id), typeID(typeID), position(position), velocity(velocity){};

    bool operator == (const Particle& rhs) const {
        return id == rhs.id;
    }
    bool operator != (const Particle& rhs) const {
        return id != rhs.id;
    }
};
