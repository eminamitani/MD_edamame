#include <iostream>
#include <string>
#include <sstream>

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
void execute_command(std::vector<Command> commands, MD& md, ThermostatType& thermostat) {
    for (const auto& cmd : commands) {
        std::cout << "コマンド: " << cmd.name << "を実行します。" << std::endl;
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
        else if (cmd.name == "NVE") {
            const RealType tsim = std::stod(args.at("duration"));
            const RealType temp = std::stod(args.at("temp"));
            const std::string output_method = args.at("output_method");
            const bool is_save_traj = args.count("trajectory") ? string_to_bool(args.at("trajectory")) : false;

            if (output_method == "log") {
                md.NVE(tsim, temp, output_method, is_save_traj);
            }
            else {
                const IntType step = std::stoi(output_method);
                md.NVE(tsim, temp, step, is_save_traj);
            }
        }
        else if (cmd.name == "NVT") {
            const RealType tsim = std::stod(args.at("duration"));
            const RealType temp = std::stod(args.at("temp"));
            const std::string output_method = args.at("output_method");
            const bool is_save_traj = args.count("trajectory") ? string_to_bool(args.at("trajectory")) : false;

            thermostat.set_temp(temp);

            if (output_method == "log") {
                md.NVT(tsim, thermostat, output_method, is_save_traj);
            }
            else {
                const IntType step = std::stoi(output_method);
                md.NVT(temp, thermostat, step, is_save_traj);
            }
        }
        else if (cmd.name == "ANNEAL") {
            const RealType cooling_rate = std::stod(args.at("cooling_rate"));
            const RealType initial_temp = std::stod(args.at("initial_temp"));
            const RealType target_temp = std::stod(args.at("target_temp"));
            const std::string output_method = args.at("output_method");
            const bool is_save_traj = args.count("trajectory") ? string_to_bool(args.at("trajectory")) : false;

            thermostat.set_temp(initial_temp);

            if (output_method == "log") {
                md.NVT_anneal(cooling_rate, thermostat, target_temp, output_method, is_save_traj);
            }
            else {
                const IntType step = std::stoi(output_method);
                md.NVT_anneal(cooling_rate, thermostat, target_temp, step, is_save_traj);
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

        if (thermostat_type == "Bussi") {
            const RealType tau = variables.count("tau") ? std::stod(variables.at("tau")) : 1.0;
            BussiThermostat thermostat(0.0, tau, device);

            execute_command(commands, md, thermostat);
        }

        if (thermostat_type == "NoseHoover") {
            const IntType chain_length = variables.count("chain_length") ? std::stoi(variables.at("chain_length")) : 1;
            const RealType tau = variables.count("tau") ? std::stod(variables.at("tau")) : 1.0;
            NoseHooverThermostat thermostat(chain_length, 0.0, tau, device);

            execute_command(commands, md, thermostat);
        }
    }

    catch (const std::exception& e) {
        std::cerr << "エラー: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}