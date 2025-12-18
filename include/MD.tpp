#include "MD.hpp"

#include "xyz.hpp"
#include "inference.hpp"
#include "config.h"
#include "LJ.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>

//=====対数サンプラー用関数=====
// ---- Sample event type (optional) ----
enum class SampleType { None, Dense, Anchor, Burst };

// ---- Burst controller: absolute-time scheduling (no off-by-one) ----
struct BurstAbs {
    int M = 0;
    int interval = 1;

    bool active = false;
    int remaining = 0;              // remaining emissions excluding anchor (idx=1..M-1)
    long long next_step = 0;        // next relative step to emit burst

    long long burst_id = 0;
    int burst_idx = 0;              // anchor=0, burst=1..M-1

    BurstAbs(int M_burst, int interval_burst)
        : M(M_burst), interval(std::max(1, interval_burst)) {}

    void trigger(long long anchor_srel) {
        ++burst_id;
        burst_idx = 0;
        remaining = M - 1;
        active = (remaining > 0);
        next_step = anchor_srel + interval;
    }

    bool step(long long s_rel) {
        if (!active) return false;
        if (s_rel == next_step) {
            ++burst_idx;
            --remaining;
            if (remaining <= 0) {
                active = false;
            } else {
                next_step += interval;
            }
            return true;
        }
        return false;
    }
};

// ---- Anchor generator: geometric anchors with ceil ----
static inline long long next_anchor_step(long long anchor_srel, long double r) {
    // guard tiny floating error before ceil
    long double x = (long double)anchor_srel * r - 1e-18L;
    return (long long) std::ceill(x);
}
// ---- Safety checker: strict condition until s_max ----

static bool is_strict_safe_until(long long t0,
                                 long double r,
                                 long long W,
                                 long long s_max)
{
    long long t = std::max(1LL, t0);

    // t が s_max を超えたら、以降は run で使わないので安全扱い
    while (t <= s_max) {
        // 次アンカー計算（オーバーフロー対策）
        long double xn = (long double)t * r - 1e-18L;
        if (xn > (long double)std::numeric_limits<long long>::max()) {
            // 次アンカーが long long に収まらない＝run範囲を超えているとみなしてOK
            return true;
        }
        long long tn = (long long)std::ceill(xn);

        // STRICT: tn - t > W
        if (tn - t <= W) return false;

        t = tn;
    }
    return true;
}

static long long find_t_safe_strict_until(int N_per_decade,
                                         int M_burst,
                                         int interval_burst,
                                         long long s_max,
                                         long long start_guess = 1,
                                         long long max_t = 1000000000000LL)
{
    if (N_per_decade < 1) throw std::invalid_argument("N_per_decade must be >= 1");
    if (M_burst < 1) throw std::invalid_argument("M_burst must be >= 1");
    if (interval_burst < 1) throw std::invalid_argument("interval_burst must be >= 1");

    const long double r = std::powl(10.0L, 1.0L / (long double)N_per_decade);
    const long long W = (long long)(M_burst - 1) * (long long)interval_burst;

    long long hi = std::max(1LL, start_guess);
    while (hi <= max_t && !is_strict_safe_until(hi, r, W, s_max)) {
        if (hi > max_t / 2) { hi = max_t + 1; break; }
        hi *= 2;
    }
    if (hi > max_t) {
        throw std::runtime_error("max_t reached in find_t_safe_strict_until; relax conditions or increase max_t.");
    }

    long long lo = hi / 2;
    while (lo + 1 < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (is_strict_safe_until(mid, r, W, s_max)) hi = mid;
        else lo = mid;
    }
    return hi;
}



// ---- Combined sampler: dense until t_safe, then anchor+burst ----
class DenseThenAnchorBurstSampler {
public:
    DenseThenAnchorBurstSampler(int N_per_decade,
                                int M_burst,
                                int interval_burst,
                                long long t_safe)
        : N_(N_per_decade),
          M_(M_burst),
          interval_(std::max(1, interval_burst)),
          t_safe_(std::max(1LL, t_safe)),
          r_(std::powl(10.0L, 1.0L / (long double)N_per_decade)),
          next_anchor_(t_safe_),
          burst_(M_burst, interval_burst)
    {
        if (N_ < 1) throw std::invalid_argument("N_per_decade must be >= 1");
        if (M_ < 1) throw std::invalid_argument("M_burst must be >= 1");
        if (interval_ < 1) throw std::invalid_argument("interval must be >= 1");
    }

    // returns: (emit?, SampleType, burst_id, burst_idx)
    std::tuple<bool, SampleType, std::optional<long long>, std::optional<int>>
    should_emit(long long s_rel)
    {
        // phase1: dense
        if (s_rel < t_safe_) {
            return {true, SampleType::Dense, std::nullopt, std::nullopt};
        }

        // phase2: anchor has priority
        if (s_rel == next_anchor_) {
            burst_.trigger(s_rel); // anchor itself is idx=0
            const long long bid = burst_.burst_id;
            next_anchor_ = next_anchor_step(s_rel, r_);
            return {true, SampleType::Anchor, bid, 0};
        }

        // phase2: burst continuation
        if (burst_.step(s_rel)) {
            return {true, SampleType::Burst, burst_.burst_id, burst_.burst_idx};
        }

        return {false, SampleType::None, std::nullopt, std::nullopt};
    }

    long long t_safe() const { return t_safe_; }
    long double r() const { return r_; }

private:
    int N_;
    int M_;
    int interval_;
    long long t_safe_;

    long double r_;
    long long next_anchor_;
    BurstAbs burst_;
};

//=====コンストラクタ=====
MD::MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device)
   : dt_(dt), NL_(cutoff, margin, device), device_(device), atoms_(Atoms(device))
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

    //その他の変数の初期化
    t_ = 0;
    temp_ = 0.0;
    dt_real_ = dt_.item<RealType>();
    box_ = torch::zeros({num_atoms_.item<IntType>(), 3}, torch::TensorOptions().dtype(kIntType).device(device_));
    traj_path_ = "./trajectory.xyz";
}

MD::MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device)
   : MD(torch::tensor(dt, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), 
        data_path, model_path, device) {}

MD::MD(RealType dt, RealType cutoff, RealType margin, const Atoms& atoms, torch::Device device) : dt_(torch::tensor(dt, torch::TensorOptions().device(device).dtype(kRealType))), device_(device), atoms_(atoms), NL_(torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), device) {
    num_atoms_ = atoms_.size();
    Lbox_ = atoms_.box_size();
    Linv_ = 1.0 / Lbox_;

    atoms_.apply_pbc();

    boltzmann_constant_ = torch::tensor(boltzmann_constant, torch::TensorOptions().dtype(kRealType).device(device_));
    conversion_factor_ = torch::tensor(conversion_factor, torch::TensorOptions().dtype(kRealType).device(device_));

    t_ = 0;
    temp_ = 0.0;
    dt_real_ = dt_.item<RealType>();
    box_ = torch::zeros({num_atoms_.item<IntType>(), 3}, torch::TensorOptions().dtype(kIntType).device(device_));
    traj_path_ = "./trajectory.xyz";
}

//=====シミュレーション=====
//保存用の関数をラムダ式で渡しているだけ

//NVEシミュレーション
void MD::NVE(const RealType tsim, const RealType temp, const IntType step, const bool is_save) {
    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);
    print_energies();

    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    if(is_save) {
        NVE_loop(tsim, temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVE_loop(tsim, temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();
            }
        });
    }
}

//NVEシミュレーション（logスケールで保存）
void MD::NVE(const RealType tsim, const RealType temp, const std::string log, const bool is_save) {
    if(log != "log") {
        return; 
    }

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);
    print_energies();

    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    const auto logbin = std::pow(10.0, 1.0 / 9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    double current_time = static_cast<double>(dt_real_) * static_cast<double>(t_);

    //現在の時間に合わせてcheckerを更新
    while (checker <= current_time) {
        checker *= logbin;
    }

    if(is_save) {
        NVE_loop(tsim, temp, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) [[unlikely]] {
                checker *= logbin;
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVE_loop(tsim, temp, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) [[unlikely]] {
                checker *= logbin;
                print_energies();
            }
        });
    }
}

//NVTシミュレーション
template <typename ThermostatType>
void MD::NVT(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save) {
    Thermostat.setup(atoms_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);
    print_energies();
    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    if(is_save) {
        NVT_loop(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVT_loop(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();
            }
        });
    }
}

// SampleType -> string
static inline const char* sample_type_str(SampleType t) {
    switch (t) {
        case SampleType::Dense:  return "dense";
        case SampleType::Anchor: return "anchor";
        case SampleType::Burst:  return "burst";
        default:                 return "none";
    }
}

// NVTシミュレーション（dense until safe, then geometric anchor + uniform burst）
template <typename ThermostatType>
void MD::NVT(const RealType tsim,
             ThermostatType& Thermostat,
             const std::string log,
             const bool is_save,
             const IntType N_per_decade,
             const IntType M_burst,
             const IntType interval_burst)
{
    if (log != "log") {
        return;
    }

    // ---- Thermostat setup ----
    Thermostat.setup(atoms_);

    // ---- Header ----
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、"
                 "total energy (eV)、temperature (K)" << std::endl;

    // ---- NL + initial inference ----
    NL_.generate(atoms_);
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    // ---- t=0 output (run start) ----
    // NOTE: t=0 も metadata を入れたいなら、emit() を呼ぶ形にしても良いです。
    print_energies();
    if (is_save) {
        // t=0 は run-relative step = 0 として扱う（必要なら）
        std::ostringstream meta0;
        meta0 << "step_rel=0"
              << " step_abs=" << static_cast<long long>(t_)
              << " time_fs=" << std::setprecision(16) << 0.0
              << " sample_type=initial";
        xyz::save_unwrapped_atoms(traj_path_, atoms_, box_, meta0.str());
    }

    // ---- Parameters ----
    const int N = (N_per_decade > 0) ? static_cast<int>(N_per_decade) : 1;
    const int M = (M_burst > 0) ? static_cast<int>(M_burst) : 1;
    const int interval = (interval_burst > 0) ? static_cast<int>(interval_burst) : 1;

    // ---- Compute strict-safe t_safe in "run-relative step" ----
    // strict: next_anchor - anchor > W, W=(M-1)*interval
    long long t_safe = 1;
    const long long nsteps = static_cast<long long>(std::llround(tsim / dt_real_));
    t_safe = find_t_safe_strict_until(N, M, interval, /*s_max=*/nsteps, /*start_guess=*/1);

    // ---- Instantiate sampler ----
    DenseThenAnchorBurstSampler sampler(N, M, interval, t_safe);

    const IntType t0 = t_; // run開始時点の通算ステップ

    // ---- Unified emission lambda ----
    auto emit = [this, is_save, t0](SampleType type,
                                    std::optional<long long> burst_id,
                                    std::optional<int> burst_idx)
    {
        // run-relative step (callback 時点の定義に合わせる)
        const long long s_rel = static_cast<long long>(t_ - t0);
        const long long s_abs = static_cast<long long>(t_);

        // time_fs を出す
        const double time_fs = static_cast<double>(s_rel) * static_cast<double>(dt_real_);

        // extxyz comment 追記用メタデータ
        std::ostringstream meta;
        meta << "step_rel=" << s_rel
             << " step_abs=" << s_abs
             << " time_fs=" << std::setprecision(16) << time_fs
             << " sample_type=" << sample_type_str(type);

        if (burst_id)  meta << " burst_id=" << *burst_id;
        if (burst_idx) meta << " burst_idx=" << *burst_idx;

        // energies のログ（従来通り）
        print_energies();

        // trajectory 保存（comment付き）
        if (is_save) {
            xyz::save_unwrapped_atoms(traj_path_, atoms_, box_, meta.str());
        }
    };

    // ---- Callback (called every MD step after t_++) ----
    auto callback = [this, t0, &sampler, &emit]() {
        const long long s_rel = static_cast<long long>(t_ - t0); // 1,2,3,...

        auto [do_emit, type, bid, bidx] = sampler.should_emit(s_rel);
        if (do_emit) [[unlikely]] {
            emit(type, bid, bidx);
        }
    };

    // ---- Run loop ----
    NVT_loop(tsim, Thermostat, callback);
}

//温度を変化させながらシミュレーション
template <typename ThermostatType>
void MD::NVT_anneal(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, const IntType step, const bool is_save) {
    // ★ ここで現在の運動温度を取得して temp_ に同期
    temp_ = atoms_.temperature().item<RealType>();
    Thermostat.set_temp(temp_);
    Thermostat.setup(atoms_);

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    print_energies();
    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    if(is_save) {
        NVT_anneal_loop(cooling_rate, Thermostat, targ_temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVT_anneal_loop(cooling_rate, Thermostat, targ_temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();
            }
        });
    }
}

// 温度変化NVTシミュレーション（dense until safe, then geometric anchor + uniform burst）
template <typename ThermostatType>
void MD::NVT_anneal(const RealType cooling_rate,
                    ThermostatType& Thermostat,
                    const RealType targ_temp,
                    const std::string log,
                    const bool is_save,
                    const IntType N_per_decade,
                    const IntType M_burst,
                    const IntType interval_burst)
{
    if (log != "log") {
        return;
    }

    // ---- Sync current kinetic temperature -> thermostat ----
    temp_ = atoms_.temperature().item<RealType>();
    Thermostat.set_temp(temp_);
    Thermostat.setup(atoms_);

    // ---- Header ----
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、"
                 "total energy (eV)、temperature (K)" << std::endl;

    // ---- NL + initial inference ----
    NL_.generate(atoms_);
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    // ---- t=0 output (run start) ----
    print_energies();
    if (is_save) {
        std::ostringstream meta0;
        meta0 << "step_rel=0"
              << " step_abs=" << static_cast<long long>(t_)
              << " time_fs=" << std::setprecision(16) << 0.0
              << " sample_type=initial";
        xyz::save_unwrapped_atoms(traj_path_, atoms_, box_, meta0.str());
    }

    // ---- Parameters ----
    const int N = (N_per_decade > 0) ? static_cast<int>(N_per_decade) : 1;
    const int M = (M_burst > 0) ? static_cast<int>(M_burst) : 1;
    const int interval = (interval_burst > 0) ? static_cast<int>(interval_burst) : 1;

    // ---- Exact anneal step count (matches your NVT_anneal_loop implementation) ----
    // 1 step あたりの下降温度
    const RealType dT = cooling_rate * dt_real_;
    // 冷却ステップ数（run-relative step の最大値として使う）
    const IntType quench_steps = static_cast<IntType>(
        std::ceil((temp_ - targ_temp) / dT)
    );
    const long long nsteps = static_cast<long long>(quench_steps);

    // ---- Compute strict-safe t_safe in "run-relative step" ----
    // strict: next_anchor - anchor > W, W=(M-1)*interval
    long long t_safe = 1;
    if (nsteps > 0) {
        t_safe = find_t_safe_strict_until(N, M, interval, /*s_max=*/nsteps, /*start_guess=*/1);
    }

    // ---- Instantiate sampler ----
    DenseThenAnchorBurstSampler sampler(N, M, interval, t_safe);

    const IntType t0 = t_; // run開始時点の通算ステップ

    // ---- Unified emission lambda ----
    auto emit = [this, is_save, t0](SampleType type,
                                    std::optional<long long> burst_id,
                                    std::optional<int> burst_idx)
    {
        const long long s_rel  = static_cast<long long>(t_ - t0);
        const long long s_abs  = static_cast<long long>(t_);
        const double time_fs   = static_cast<double>(s_rel) * static_cast<double>(dt_real_);

        std::ostringstream meta;
        meta << "step_rel=" << s_rel
             << " step_abs=" << s_abs
             << " time_fs=" << std::setprecision(16) << time_fs
             << " sample_type=" << sample_type_str(type);

        if (burst_id)  meta << " burst_id=" << *burst_id;
        if (burst_idx) meta << " burst_idx=" << *burst_idx;

        // energies のログ（従来通り）
        print_energies();

        // trajectory 保存（comment付き）
        if (is_save) {
            xyz::save_unwrapped_atoms(traj_path_, atoms_, box_, meta.str());
        }
    };

    // ---- Callback (called every MD step after t_++) ----
    auto callback = [this, t0, &sampler, &emit]() {
        const long long s_rel = static_cast<long long>(t_ - t0); // 1,2,3,...

        auto [do_emit, type, bid, bidx] = sampler.should_emit(s_rel);
        if (do_emit) [[unlikely]] {
            emit(type, bid, bidx);
        }
    };

    // ---- Run loop (temperature schedule handled inside) ----
    NVT_anneal_loop(cooling_rate, Thermostat, targ_temp, callback);
}


//=====シミュレーション（1ステップ）=====
//NVEの1ステップ
void MD::step() {
    atoms_.velocities_update(dt_);      //速度の更新（1回目）
    atoms_.positions_update(dt_, box_);  //位置の更新
    NL_.update(atoms_);                 //NLの確認と更新
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
    atoms_.velocities_update(dt_);      //速度の更新（2回目）
}

//NVTの1ステップ（Nose-Hoover）
void MD::step(NoseHooverThermostat& Thermostat) {
    Thermostat.update(atoms_, dt_);     //熱浴の更新
    step();
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

//NVTの1ステップ（Bussi）
void MD::step(BussiThermostat& Thermostat) {
    step();
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

//=====シミュレーション（メインループ）=====
template <typename OutputAction>
void MD::NVE_loop(const RealType tsim, const RealType temp, OutputAction output_action) {
    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;

    while(t_ < steps){
        step();

        t_ ++;

        //出力
        output_action();
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_loop(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action) {
    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;

    while(t_ < steps){
        step(Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_anneal_loop(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action) {
    RealType dT = cooling_rate * dt_real_;                                         //1ステップあたりの下降温度
    IntType quench_steps = static_cast<IntType>(std::ceil((temp_ - targ_temp) / dT));    //冷却ステップ数

    if (temp_ < targ_temp) {
        quench_steps = -quench_steps;
        dT = -dT;
    }

    quench_steps += t_;

    //冷却
    //ANNEALの場合はここでtemp_による制御が入っているから、
    //最初の段階でtemp_を初期化する必要があった。
    while(t_ < quench_steps){
        temp_ -= dT;
        Thermostat.set_temp(temp_);
        step(Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }

    temp_ = targ_temp;
    Thermostat.set_temp(targ_temp);
}

//=====その他=====
//速度（温度）の初期化
void MD::init_temp(const RealType initial_temp){
    temp_ = initial_temp;

    //平均0、分散1のランダムな分布を作成
    torch::Tensor velocities = torch::randn({num_atoms_.item<int64_t>(), 3}, torch::TensorOptions().device(device_).dtype(kRealType));

    //分散を√(k_B * T / m)にする。
    //この時、(eV / amu) -> ((Å / fs) ^ 2)
    torch::Tensor masses = atoms_.masses();
    torch::Tensor temp = torch::tensor(temp_, torch::TensorOptions().dtype(kRealType).device(device_));
    torch::Tensor sigma = torch::sqrt((boltzmann_constant_ * temp_ * conversion_factor_) / masses);
    //velocitiesにsigmaを掛けることで分散を調節。
    //この時、velocities (N, 3)とsigma (N, )を計算するために、sigma (N, ) -> (N, 1)
    velocities *= sigma.unsqueeze(1);

    //全体速度の除去
    torch::Tensor drift_velocity = torch::mean(velocities, 0);
    velocities -= drift_velocity;

    atoms_.set_velocities(velocities);
}

//エネルギーの出力
void MD::print_energies(){
    RealType K = atoms_.kinetic_energy().item<RealType>();
    RealType U = atoms_.potential_energy().item<RealType>();
    RealType temperature = atoms_.temperature().item<RealType>();
    
    //時刻、運動エネルギー、ポテンシャルエネルギー、全エネルギー、温度を出力
    std::cout << std::setprecision(15) << std::scientific << dt_.item<RealType>() * t_ << "," 
                                                          << K << "," 
                                                          << U << "," 
                                                          << K + U << "," 
                                                          << temperature << std::endl;
}

void MD::reset_step() {
    t_ = 0;
}

void MD::reset_box() {
    box_ = torch::zeros({num_atoms_.item<IntType>(), 3}, torch::TensorOptions().dtype(kIntType).device(device_));
}

void MD::save_atoms(const std::string& save_path) {
    xyz::save_atoms(save_path, atoms_);
}

void MD::save_unwrapped_atoms(const std::string& save_path) {
    xyz::save_unwrapped_atoms(save_path, atoms_, box_);
}

void MD::set_traj_path(const std::string& path) {
    traj_path_ = path;
}

//=====LJユニットによるテスト用関数=====
//NVEの1ステップ
void MD::step_LJ(torch::Tensor& box) {
    atoms_.velocities_update(dt_);      //速度の更新（1回目）
    atoms_.positions_update(dt_, box);  //位置の更新
    NL_.update(atoms_);                 //NLの確認と更新
    LJ::calc_energy_and_force(atoms_, NL_); //力の更新
    atoms_.velocities_update(dt_);      //速度の更新（2回目）
}

//NVTの1ステップ（Nose-Hoover）
void MD::step_LJ(torch::Tensor& box, NoseHooverThermostat& Thermostat) {
    Thermostat.update(atoms_, dt_);     //熱浴の更新
    step_LJ(box);
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

//NVTの1ステップ（Bussi）
void MD::step_LJ(torch::Tensor& box, BussiThermostat& Thermostat) {
    step_LJ(box);
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

template <typename OutputAction>
void MD::NVE_loop_LJ(const RealType tsim, const RealType temp, OutputAction output_action) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    LJ::calc_energy_and_force(atoms_, NL_);

    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;
    print_energies();

    while(t_ < steps){
        step_LJ(box_);

        t_ ++;

        //出力
        output_action();
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_loop_LJ(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    Thermostat.setup(atoms_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    LJ::calc_energy_and_force(atoms_, NL_);

    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;
    print_energies();

    while(t_ < steps){
        step_LJ(box_, Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_anneal_loop_LJ(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    Thermostat.setup(atoms_);

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    LJ::calc_energy_and_force(atoms_, NL_);

    const RealType dT = cooling_rate * dt_real_;                                            //1ステップあたりの下降温度
    IntType quench_steps = static_cast<IntType>(std::ceil((temp_ - targ_temp) / dT));       //冷却ステップ数
    quench_steps += t_;

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    print_energies();

    //冷却
    while(t_ < quench_steps){
        temp_ -= dT;
        Thermostat.set_temp(temp_);
        step_LJ(box_, Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }

    temp_ = targ_temp;
    Thermostat.set_temp(targ_temp);
}

//NVEシミュレーション
void MD::NVE_LJ(const RealType tsim, const RealType temp, const IntType step, const bool is_save, const std::string output_path) {
    if(is_save) {
        NVE_loop_LJ(tsim, temp, [this, step]() {
            if(t_ % step == 0) {
                print_energies();

                std::string path = "./outputs/output_traj.xyz";
                xyz::save_unwrapped_atoms(path, atoms_, box_);
            }
        });
    }
    else {
        NVE_loop_LJ(tsim, temp, [this, step]() {
            if(t_ % step == 0) {
                print_energies();
            }
        });
    }

    xyz::save_atoms(output_path, atoms_);
}

//NVTシミュレーション
template <typename ThermostatType>
void MD::NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save, const std::string output_path) {
    if(is_save) {
        NVT_loop_LJ(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) {
                print_energies();

                std::string path = "./outputs/output_traj.xyz";
                xyz::save_unwrapped_atoms(path, atoms_, box_);
            }
        });
    }
    else {
        NVT_loop_LJ(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) {
                print_energies();
            }
        });
    }

    xyz::save_atoms(output_path, atoms_);
}

//NVTシミュレーション（logスケールで保存）
template <typename ThermostatType>
void MD::NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const std::string log, const bool is_save, const std::string output_path) {
    if(log != "log") {
        return; 
    }

    const auto logbin = std::pow(10.0, 1.0 / 9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    if(is_save) {
        NVT_loop_LJ(tsim, Thermostat, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) {
                checker *= logbin;
                print_energies();

                std::string path = "./outputs/output_traj.xyz";
                xyz::save_unwrapped_atoms(path, atoms_, box_);
            }
        });
    }
    else {
        NVT_loop_LJ(tsim, Thermostat, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) {
                checker *= logbin;
                print_energies();
            }
        });
    }

    xyz::save_atoms(output_path, atoms_);
}
