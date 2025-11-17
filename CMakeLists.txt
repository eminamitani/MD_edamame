cmake_minimum_required(VERSION 3.10)
project(MD_MLP)

# C++17 を使用（必要に応じて変更）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# libtorch via conda
set(LIBTORCH_PATH "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/torch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# cuDNN via conda
set(CUDNN_ROOT "$ENV{CONDA_PREFIX}")

find_path(CUDNN_INCLUDE_DIR cudnn.h
          PATHS ${CUDNN_ROOT}/include)

find_library(CUDNN_LIBRARY cudnn
             PATHS ${CUDNN_ROOT}/lib ${CUDNN_ROOT}/lib64)

if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    set(USE_CUDNN 1)
else()
    message(WARNING "cuDNN not found. Compiling without cuDNN support.")
    set(USE_CUDNN 0)
endif()

# ソースファイルのリスト
set(SOURCES
    src/main.cpp
    src/NoseHooverThermostat.cpp
    src/BussiThermostat.cpp
    src/Atoms.cpp
    src/Atom.cpp
    src/NeighbourList.cpp
    src/xyz.cpp
    src/inference.cpp
    src/LJ.cpp
    src/ConfigReader.cpp
)

# 実行ファイルを作成
add_executable(MD_MLP ${SOURCES})

# libtorch + cuDNN をリンク
target_link_libraries(MD_MLP ${TORCH_LIBRARIES} ${CUDNN_LIBRARY})

target_compile_options(MD_MLP PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-O3,-fPIC
    >
)

set_target_properties(MD_MLP PROPERTIES CUDA_ARCHITECTURES 89)

set_property(TARGET MD_MLP PROPERTY POSITION_INDEPENDENT_CODE ON)

