#ifndef MD_HPP
#define MD_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"
#include "config.h"
#include "NoseHooverThermostat.hpp"
#include "BussiThermostat.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <optional>

class MD{
    public:
        //コンストラクタ
        MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device = torch::kCPU); 
        MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device = torch::kCPU); 
        MD(RealType dt, RealType cutoff, RealType margin, const Atoms& atoms, torch::Device device = torch::kCPU); 

        //シミュレーション
        void NVE(const RealType tsim, const RealType temp, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");
        void NVE(const RealType tsim, const RealType temp, const std::string log, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");

        //一定温度のシミュレーション
        template <typename ThermostatType>
        void NVT(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");            
        template <typename ThermostatType>
        void NVT(const RealType tsim, ThermostatType& Thermostat, const std::string log, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");        

        //温度変化をさせるシミュレーション
        template <typename ThermostatType>
        void NVT_anneal(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");
        template <typename ThermostatType>
        void NVT_anneal(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, const std::string log, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");

        //原子の保存
        void save_atoms(const std::string& save_path);
        void save_unwrapped_atoms(const std::string& save_path);

        //時間のリセット（桁溢れ対策）
        void reset_step();

        //テスト用
        void NVE_LJ(const RealType tsim, const RealType temp, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");

        template <typename ThermostatType>
        void NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const std::string log, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");        

        template <typename ThermostatType>
        void NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");        


    private:
        //その他（補助用関数）
        void print_energies();                                          //結果の出力
        void init_temp(const RealType initial_temp);                     //原子の速度（温度）の初期化

        void step(torch::Tensor& box);                                  //1ステップ
        void step(torch::Tensor& box, NoseHooverThermostat& Thermostat);
        void step(torch::Tensor& box, BussiThermostat& Thermostat);

        template <typename OutputAction>
        void NVE_loop(const RealType tsim, const RealType temp, OutputAction output_action);

        template <typename OutputAction, typename ThermostatType>
        void NVT_loop(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action);

        template <typename OutputAction, typename ThermostatType>
        void NVT_anneal_loop(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action);

        //テスト用
        void step_LJ(torch::Tensor& box);                                  //1ステップ
        void step_LJ(torch::Tensor& box, NoseHooverThermostat& Thermostat);
        void step_LJ(torch::Tensor& box, BussiThermostat& Thermostat);

        template <typename OutputAction>
        void NVE_loop_LJ(const RealType tsim, const RealType temp, OutputAction output_action);

        template <typename OutputAction, typename ThermostatType>
        void NVT_loop_LJ(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action);

        template <typename OutputAction, typename ThermostatType>
        void NVT_anneal_loop_LJ(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action);

        //シミュレーション用
        IntType t_;                                                     //現在のステップ数
        RealType temp_;                                                 //現在の温度
        torch::Tensor dt_;                                              //時間刻み幅
        RealType dt_real_;
        torch::Tensor Lbox_;                                            //シミュレーションセルのサイズ
        torch::Tensor Linv_;                                            //セルのサイズの逆数
        NeighbourList NL_;                                              //隣接リスト

        torch::Tensor box_;                                             //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)

        //MLP用変数
        torch::jit::script::Module module_;                              //モデルを格納する変数

        //系
        Atoms atoms_;                                                    //原子
        torch::Tensor num_atoms_;                                        //原子数

        //シミュレーションデバイス
        torch::Device device_;

        //定数
        torch::Tensor boltzmann_constant_;
        torch::Tensor conversion_factor_;
};

#include "MD.tpp"

#endif