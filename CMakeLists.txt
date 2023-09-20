# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Project's name
project(ANNC LANGUAGES CXX)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "No CUDA compiler found")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

set(EXECUTABLE_OUTPUT_PATH "bin")
set(LIBRARY_OUTPUT_PATH "lib")


add_library(nn_lib STATIC
  include/nn/NeuralNetwork.hpp
  )

add_library(fterm_lib STATIC
  include/fterm/FTerm.hpp
  )

add_library(domain_lib STATIC
  include/domain/Domain.hpp
  include/domain/Rectangle.hpp
  include/domain/TwoDimDomain.hpp
  )


add_library(utils_lib STATIC
  include/utils/ActivationFunctions.hpp
  src/utils/ActivationFunctions.cpp
  include/utils/ReadFile.hpp
  src/utils/ReadFile.cpp
  include/utils/WriteFile.hpp
  src/utils/WriteFile.cpp
  include/utils/Gen2DData.hpp
  src/utils/Gen2DData.cpp
  )

add_library(optimizer_lib STATIC
  include/optimizer/Adam.hpp
  include/optimizer/Optimizer.hpp  
  )

add_library(scheduler_lib STATIC
  include/scheduler/Scheduler.hpp
  )


target_include_directories(nn_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_include_directories(utils_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_include_directories(domain_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_include_directories(fterm_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_include_directories(optimizer_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_include_directories(scheduler_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )

# target_include_directories(gpu_lib PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )


target_compile_options(nn_lib  PRIVATE "-w")
target_compile_features(nn_lib  PUBLIC cxx_std_11  )
target_compile_options(utils_lib  PRIVATE "-w")
target_compile_features(utils_lib  PUBLIC cxx_std_11 )
target_compile_options(domain_lib  PRIVATE "-w")
target_compile_features(domain_lib  PUBLIC cxx_std_11 )
target_compile_options(fterm_lib  PRIVATE "-w")
target_compile_features(fterm_lib  PUBLIC cxx_std_11 )
target_compile_options(optimizer_lib  PRIVATE "-w")
target_compile_features(optimizer_lib  PUBLIC cxx_std_11 )
target_compile_options(scheduler_lib  PRIVATE "-w")
target_compile_features(scheduler_lib  PUBLIC cxx_std_11 )


set_target_properties( nn_lib  PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties( utils_lib  PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties( domain_lib  PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties( fterm_lib  PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties( optimizer_lib  PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties( scheduler_lib  PROPERTIES LINKER_LANGUAGE CXX)
# set_target_properties( gpu_lib  PROPERTIES CUDA_SEPARABLE_COMPILATION ON )



# add_executable(example1 examples/example1 )
# set_target_properties(example1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)        
# target_link_libraries(example1 PRIVATE nn_lib utils_lib)
# target_include_directories(example1 PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
# target_compile_options(example1  PRIVATE "-w")

# add_executable(matrix_cpp examples/matrix/matrix.cpp
#   include/num/Matrix.hpp
#   include/num/MatrixAdd.hpp
#   include/num/MatrixScalarMult.hpp
#   include/num/MatrixMultMult.hpp
#   include/num/TOperators.hpp )


# add_executable(example1_cpp examples/example1/example1.cpp )
# target_link_libraries(example1_cpp PRIVATE  nn_lib utils_lib  )
# target_include_directories(example1_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
# target_compile_options(example1_cpp  PRIVATE "-w")



# add_executable(example1_cpp examples/example1/example1.cpp )
# target_link_libraries(example1_cpp PRIVATE nn_lib utils_lib  )
# target_include_directories(example1_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
# target_compile_options(example1_cpp  PRIVATE "-w")

# add_executable(example2_cpp examples/example2/example2.cpp )
# target_link_libraries(example2_cpp PRIVATE nn_lib utils_lib  )
# target_include_directories(example2_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
# target_compile_options(example2_cpp  PRIVATE "-w")


# add_executable(example3_cpp examples/example3/example3.cpp )
# target_link_libraries(example3_cpp PRIVATE  nn_lib utils_lib domain_lib utils_lib)
# target_include_directories(example3_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
# target_compile_options(example3_cpp  PRIVATE "-w")

# add_executable(example4_cpp examples/example4/example4.cpp )
# target_link_libraries(example4_cpp PRIVATE  nn_lib utils_lib domain_lib utils_lib)
# target_include_directories(example4_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
# target_compile_options(example4_cpp  PRIVATE "-w")


add_executable(tCClassification_cpp examples/tCClassification/tCClassification.cpp )
# target_link_libraries(tCClassification_cpp PRIVATE  nn_lib utils_lib domain_lib utils_lib fterm_lib)
target_include_directories(tCClassification_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_compile_options(tCClassification_cpp  PRIVATE "-w")

add_executable(poisson2D_cpp examples/poisson2D/poisson2D.cpp )
# target_link_libraries(poisson2D_cpp PRIVATE  nn_lib utils_lib domain_lib utils_lib fterm_lib)
target_include_directories(poisson2D_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_compile_options(poisson2D_cpp  PRIVATE "-w")


add_executable(vectorfield2D_cpp examples/vectorfield2D/vectorfield2D.cpp )
# target_link_libraries(vectorfield2D_cpp PRIVATE  nn_lib utils_lib domain_lib utils_lib fterm_lib)
target_include_directories(vectorfield2D_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_compile_options(vectorfield2D_cpp  PRIVATE "-w")


add_executable(interp1D_cpp examples/interp1D/interp1D.cpp )
# target_link_libraries(interp1D_cpp PRIVATE  nn_lib utils_lib domain_lib utils_lib fterm_lib optimizer_lib scheduler_lib)
target_include_directories(interp1D_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/include/Eigen )
target_compile_options(interp1D_cpp  PRIVATE "-w")