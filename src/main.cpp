#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

#include "Command.hpp"
#include "ConfigReader.hpp"
#include "MD.hpp"

//文字列をboolに変換
bool string_to_bool(const std::string& s) {
    std::stringstream ss(s);
    bool b;
    ss >> std::boolalpha >> b;
    return b;
}

//コマンドの実行
template <typename ThermostatType>
void execute_command(std::vector<Command> commands, MD& md, ThermostatType& thermostat, const RealType& dt) {
    for (const auto& cmd : commands) {
        std::cout << "=====" << cmd.name << "=====" << std::endl;
        auto args = cmd.args;
        if (cmd.name == "SAVE") {
            std::string output_path = args.count("output") ? args.at("output") : "saved_structure.xyz";
            md.save_atoms(output_path);
            std::cout << output_path << "に構造を保存しました。" << std::endl;
        }
        else if (cmd.name == "SAVE_UNWRAPPED") {
            std::string output_path = args.count("output") ? args.at("output") : "saved_structure.xyz";
            md.save_unwrapped_atoms(output_path);
            std::cout << output_path << "にアンラップした構造を保存しました。" << std::endl;
        }
        else if (cmd.name == "RESET_STEP") {
            md.reset_step();
            std::cout << "ステップ数を0に初期化しました。" << std::endl;
        }
        else if (cmd.name == "INIT_TEMP") {
            const RealType temp = std::stod(args.at("temp"));
            md.init_temp(temp);
            std::cout << "温度を" << temp << " Kで初期化しました。" << std::endl;
        }
        else if (cmd.name == "NVE") {
            const RealType tsim = std::stod(args.at("duration"));
            const RealType temp = std::stod(args.at("temp"));
            const std::string output_method = args.at("output_method");
            const bool is_save_traj = args.count("trajectory") ? string_to_bool(args.at("trajectory")) : false;

            std::cout << "シミュレーション時間: " << tsim << " fs\n"
                      << "ステップ数: " << tsim / dt << "\n"
                      << "初期温度: " << temp << " K\n"
                      << "保存間隔: " << output_method << "\n"
                      << "トラジェクトリの保存: " << is_save_traj << std::endl;

            if (output_method == "log") {
                md.NVE(tsim, temp, output_method, is_save_traj);
            }
            else {
                const IntType step = std::stoi(output_method);
                md.NVE(tsim, temp, step, is_save_traj);
            }

            const std::string& save_path = cmd.redirect_target;
            if (!save_path.empty()) {
                md.save_atoms(save_path);
                std::cout << save_path << "に構造を保存しました。" << std::endl;
            }
        }
        else if (cmd.name == "NVT") {
            const RealType tsim = std::stod(args.at("duration"));
            const RealType temp = std::stod(args.at("temp"));
            const std::string output_method = args.at("output_method");
            const bool is_save_traj = args.count("trajectory") ? string_to_bool(args.at("trajectory")) : false;

            thermostat.set_temp(temp);

            std::cout << "シミュレーション時間: " << tsim << " fs\n"
                      << "ステップ数: " << tsim / dt << "\n"
                      << "温度: " << temp << " K\n"
                      << "保存間隔: " << output_method << "\n"
                      << "トラジェクトリの保存: " << is_save_traj << std::endl;

            if (output_method == "log") {
                md.NVT(tsim, thermostat, output_method, is_save_traj);
            }
            else {
                const IntType step = std::stoi(output_method);
                md.NVT(temp, thermostat, step, is_save_traj);
            }

            const std::string& save_path = cmd.redirect_target;
            if (!save_path.empty()) {
                md.save_atoms(save_path);
                std::cout << save_path << "に構造を保存しました。" << std::endl;
            }
        }
        else if (cmd.name == "ANNEAL") {
            const RealType cooling_rate = std::stod(args.at("cooling_rate"));
            const RealType initial_temp = std::stod(args.at("initial_temp"));
            const RealType target_temp = std::stod(args.at("target_temp"));
            const std::string output_method = args.at("output_method");
            const bool is_save_traj = args.count("trajectory") ? string_to_bool(args.at("trajectory")) : false;

            std::cout << "冷却速度: " << cooling_rate << " K/fs\n"
                      << "初期温度: " << initial_temp << " K\n"
                      << "目標温度: " << target_temp << " K\n"
                      << "ステップ数: " << static_cast<IntType>((initial_temp - target_temp) / (cooling_rate * dt)) << "\n"
                      << "保存間隔: " << output_method << "\n"
                      << "トラジェクトリの保存: " << is_save_traj << std::endl;

            thermostat.set_temp(initial_temp);

            if (output_method == "log") {
                md.NVT_anneal(cooling_rate, thermostat, target_temp, output_method, is_save_traj);
            }
            else {
                const IntType step = std::stoi(output_method);
                md.NVT_anneal(cooling_rate, thermostat, target_temp, step, is_save_traj);
            }

            const std::string& save_path = cmd.redirect_target;
            if (!save_path.empty()) {
                md.save_atoms(save_path);
                std::cout << save_path << "に構造を保存しました。" << std::endl;
            }
        }
        else {
            std::cerr << "未知のコマンド: " << cmd.name << "をスキップしました。" << std::endl;
            continue;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "設定ファイルを指定してください。" << std::endl;
        return 1;
    }

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    try {
        //設定の読み込み
        ConfigReader reader(argv[1]);
        auto commands = reader.commands();
        auto variables = reader.variables();

        //変数の設定
        const RealType dt = variables.count("dt") ? std::stod(variables.at("dt")) : 0.5;
        const RealType cutoff = variables.count("cutoff") ? std::stod(variables.at("cutoff")) : 5.0;
        const RealType margin = variables.count("margin") ? std::stod(variables.at("margin")) : 1.0;

        const std::string trajectory_path = variables.count("trajectory_path") ? variables.at("trajectory_path") : "./trajectory.xyz";
        const std::string thermostat_type = variables.count("thermostat_type") ? variables.at("thermostat_type") : "Bussi";

        //必須の値
        const std::string model_path = variables.at("model_path");
        const std::string initial_path = variables.at("initial_path");

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        MD md(dt, cutoff, margin, initial_path, model_path, device);

        md.set_traj_path(trajectory_path);

        //設定を出力
        std::cout << "=====全体の設定=====" << std::endl 
                  << "初期構造: " << initial_path << std::endl
                  << "モデル: " << model_path << std::endl
                  << "タイムステップ: " << dt << " fs" << std::endl
                  << "カットオフ距離: " << cutoff << " Å" << std::endl
                  << "マージン: " << margin << " Å" << std::endl
                  << "熱浴の種類: " << thermostat_type << std::endl;

        std::cout << "=====出力設定=====" << std::endl
                  << "トラジェクトリの保存先: " << trajectory_path << std::endl;

        if (thermostat_type == "Bussi") {
            const RealType tau = variables.count("tau") ? std::stod(variables.at("tau")) : 1.0;
            BussiThermostat thermostat(0.0, tau, device);

            execute_command(commands, md, thermostat, dt);
        }

        if (thermostat_type == "NoseHoover") {
            const IntType chain_length = variables.count("chain_length") ? std::stoi(variables.at("chain_length")) : 1;
            const RealType tau = variables.count("tau") ? std::stod(variables.at("tau")) : dt * 1e+3;
            NoseHooverThermostat thermostat(chain_length, 0.0, tau, device);

            execute_command(commands, md, thermostat, dt);
        }
    }

    catch (const std::exception& e) {
        std::cerr << "エラー: " << e.what() << std::endl;
        return 1;
    }

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;

    return 0;
}