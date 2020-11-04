
#include "Simulation.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Simulation simulation = Simulation(argc, argv);
    simulation.start();
  }
  Kokkos::finalize();
  return 0;
}
