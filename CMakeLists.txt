cmake_minimum_required(VERSION 3.10)
project(MD_MLP)

# C++17 を使用（必要に応じて変更）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# libtorchの場所
set(LIBTORCH_PATH "/home/nozawa/mambaforge/envs/md/lib/python3.13/site-packages/torch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# cuDNNの設定
set(CUDNN_ROOT "/home/nozawa/mambaforge/envs/md/") 
find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDNN_ROOT}/include /usr/include)
find_library(CUDNN_LIBRARY cudnn PATHS ${CUDNN_ROOT}/lib64 ${CUDNN_ROOT}/lib /usr/lib/x86_64-linux-gnu)

if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    include_directories(${CUDNN_INCLUDE_DIR})
    add_definitions(-DUSE_CUDNN=1)
else()
    message(WARNING "cuDNN not found. Compiling without cuDNN support.")
    add_definitions(-DUSE_CUDNN=0)
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

# cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc ..