cmake_minimum_required(VERSION 3.18)
project(test_libqe CXX)

find_package(libqe       REQUIRED)
find_package(Armadillo   REQUIRED)

add_executable(test_libqe src/test_libqe.cpp)

target_link_libraries(test_libqe PRIVATE
    libqe::libqe
    ${ARMADILLO_LIBRARIES}
)

target_include_directories(test_libqe PRIVATE ${ARMADILLO_INCLUDE_DIRS})
