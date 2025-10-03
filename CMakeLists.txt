cmake_minimum_required(VERSION 3.10)
project(MD_MLP)
#project(MD_LJ)

# C++17 を使用（必要に応じて変更）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# libtorchの場所
set(LIBTORCH_PATH "/home/nozawa/mambaforge/envs/md/lib/python3.13/site-packages/torch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")

# libtorchのCMakeモジュールを探す
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# ソースファイルのリスト
set(SOURCES
    src/main.cpp
#    src/main_NVT_MSD.cpp
#    src/main_LJ_NoseHoover.cpp
#    src/simulate_LJ_NHC_torch_4.cpp
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
#add_executable(MD_LJ ${SOURCES})

# libtorchをリンク
target_link_libraries(MD_MLP ${TORCH_LIBRARIES})
#target_link_libraries(MD_LJ ${TORCH_LIBRARIES})

# 必須：位置独立コード（libtorchで必要）
set_property(TARGET MD_MLP PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_property(TARGET MD_LJ PROPERTY POSITION_INDEPENDENT_CODE ON)

# cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc ..