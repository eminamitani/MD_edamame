#include "MD.hpp"

#include "xyz.hpp"
#include "inference.hpp"
#include "config.h"
#include "LJ.hpp"

//コンストラクタ
MD::MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device)
   : dt_(dt), atoms_(device), NL_(cutoff, margin, device), device_(device)
{
    //モデルの読み込み
    module_ = inference::load_model(model_path);
    module_.to(device);

    //初期構造のロード
    xyz::load_atoms(data_path, atoms_, device);
    num_atoms_ = atoms_.size();
    Lbox_ = atoms_.box_size();
    Linv_ = 1.0 / Lbox_;

    //周期境界条件の補正
    atoms_.apply_pbc();

    //使用する定数のデバイスを移動しておく。
    boltzmann_constant_ = torch::tensor(boltzmann_constant, torch::TensorOptions().dtype(kRealType).device(device_));
    conversion_factor_ = torch::tensor(conversion_factor, torch::TensorOptions().dtype(kRealType).device(device_));
}

MD::MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device)
   : MD(torch::tensor(dt, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), 
        data_path, model_path, device) {}

MD::MD(RealType dt, RealType cutoff, RealType margin, Atoms atoms, torch::Device device) : device_(device), atoms_(atoms), NL_(torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), device) {
    num_atoms_ = atoms_.size();
    Lbox_ = atoms_.box_size();
    Linv_ = 1.0 / Lbox_;

    atoms_.apply_pbc();

    boltzmann_constant_ = torch::tensor(boltzmann_constant, torch::TensorOptions().dtype(kRealType).device(device_));
    conversion_factor_ = torch::tensor(conversion_factor, torch::TensorOptions().dtype(kRealType).device(device_));
}

//速度の初期化
void MD::init_vel_MB(const RealType float_targ){
    //平均0、分散1のランダムな分布を作成
    torch::Tensor velocities = torch::randn({num_atoms_.item<int64_t>(), 3}, torch::TensorOptions().device(device_).dtype(kRealType));

    //分散を√(k_B * T / m)にする。
    //この時、(eV / amu) -> ((Å / fs) ^ 2)
    torch::Tensor masses = atoms_.masses();
    torch::Tensor temp = torch::tensor(float_targ, torch::TensorOptions().dtype(kRealType).device(device_));
    torch::Tensor sigma = torch::sqrt((boltzmann_constant_ * float_targ * conversion_factor_) / masses);
    //velocitiesにsigmaを掛けることで分散を調節。
    //この時、velocities (N, 3)とsigma (N, )を計算するために、sigma (N, ) -> (N, 1)
    velocities *= sigma.unsqueeze(1);

    //全体速度の除去
    torch::Tensor drift_velocity = torch::mean(velocities, 0);
    velocities -= drift_velocity;

    atoms_.set_velocities(velocities);
}

//シミュレーション
void MD::NVE(const RealType tsim) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    const long steps = tsim / dt_.item<RealType>();    //総ステップ数
    print_energies(t);

    while(t < steps){
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）

        t ++;

        //出力
        //とりあえず100ステップごとに出力
        if(t % 100 == 0){
            print_energies(t);
        }
    }
}

//シミュレーション
void MD::NVE_log(const RealType tsim) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    const long steps = tsim / dt_.item<RealType>();    //総ステップ数
    print_energies(t);

    //ログスケールでの出力のための変数
    const auto logbin = std::pow(10.0, 1.0 / 9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    while(t < steps){
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）

        t ++;

        //出力
        //ログスケールで出力
        if(dt_.item<RealType>() * t > checker){
            checker *= logbin;
            print_energies(t);
        }
    }
}

void MD::NVE_from_grad(const RealType tsim){
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    const long steps = tsim / dt_.item<RealType>();    //総ステップ数
    print_energies(t);

    while(t < steps){
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::infer_energy_with_MLP_and_clac_force(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）

        t ++;

        //出力
        //とりあえず100ステップごとに出力
        if(t % 100 == 0){
            print_energies(t);
        }
    }
}

//シミュレーション後、構造を保存
void MD::NVE_save(const RealType tsim){
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    const long steps = tsim / dt_.item<RealType>();    //総ステップ数
    print_energies(t);

    while(t < steps){
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）

        t ++;

        //出力
        //とりあえず100ステップごとに出力
        if(t % 100 == 0){
            print_energies(t);
        }
    }

    std::string save_path = "data/saved_structure.xyz";
    xyz::save_atoms(save_path, atoms_);
}

//Nose-Hoover Thermostatを用いたNVTシミュレーション
void MD::NVT(const RealType tsim, NoseHooverThermostat& Thermostat) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    Thermostat.setup(atoms_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    RealType dt_real = dt_.item<RealType>();
    const long steps = tsim / dt_real;    //総ステップ数
    print_energies(t);

    while(t < steps){
        Thermostat.update(atoms_, dt_);
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）
        Thermostat.update(atoms_, dt_);

        t ++;

        //出力
        //とりあえず100ステップごとに出力
        if(t % 100 == 0){
            print_energies(t);
        }
    }
}

//Bussi Thermostatを用いたNVTシミュレーション
void MD::NVT(const RealType tsim, BussiThermostat& Thermostat) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    Thermostat.setup(atoms_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    RealType dt_real = dt_.item<RealType>();
    const long steps = tsim / dt_real;    //総ステップ数
    print_energies(t);

    while(t < steps){
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）
        Thermostat.update(atoms_, dt_);

        t ++;

        //出力
        //とりあえず100ステップごとに出力
        if(t % 100 == 0){
            print_energies(t);
        }
    }
}

//-----補助用関数-----
//エネルギーの出力
void MD::print_energies(long t){
    RealType K = atoms_.kinetic_energy().item<RealType>();
    RealType U = atoms_.potential_energy().item<RealType>();
    RealType temperature = atoms_.temperature().item<RealType>();
    
    //時刻、1粒子当たりの運動エネルギー、1粒子当たりのポテンシャルエネルギー、1粒子当たりの全エネルギーを出力
    std::cout << std::setprecision(15) << std::scientific << dt_.item<RealType>() * t << "," 
                                                          << K << "," 
                                                          << U << "," 
                                                          << K + U << "," 
                                                          << temperature << std::endl;
}