#include <iostream>
#include <Kokkos_Core.hpp>

int main (int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    Kokkos::parallel_for( "test", 1000, KOKKOS_LAMBDA ( int i ) {
        std::cout << i << std::endl;
    });
    Kokkos::finalize();
    return 0;
}
