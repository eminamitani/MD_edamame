#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ConfigReader.hpp"

//ヘルパー関数
namespace {
    //空白を除去
    std::string trim(const std::string& str) {
        const std::string whitespace = " \t\n\r";
        //最初の空白
        const auto strBegin = str.find_first_not_of(whitespace);
        if (strBegin == std::string::npos) return "";
        //最後の空白
        const auto strEnd = str.find_last_not_of(whitespace);
        return str.substr(strBegin, strEnd - strBegin + 1);
    }

    //分割
    std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;

        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }

        return tokens;
    }

    //変数をその中身に置き換え
    std::string substitute_vars(std::string value, std::map<std::string, std::string> variables) {
        for (const auto& pair : variables) {
            std::string var_key = "${" + pair.first + "}";
            std::size_t start_pos = 0;
            //文字列から、"${変数名}"を探して、変数の値に置き換える
            while ((start_pos = value.find(var_key, start_pos)) != std::string::npos) {
                value.replace(start_pos, var_key.length(), pair.second);
                start_pos += pair.second.length();
            }
        }
        return value;
    }
}

//コンストラクタ（parse関数を呼ぶだけ）
ConfigReader::ConfigReader(const std::string& filename) {
    parse(filename);
}

void ConfigReader::parse(const std::string& filename) {
    std::ifstream file(filename);

    if(!file.is_open()) {
        throw std::runtime_error("設定ファイルを開けませんでした。");
    }

    std::vector<std::string> lines;
    std::string line;

    //ファイルを1行ずつ読み込む
    while(std::getline(file, line)) {
        lines.push_back(line);
    }

    //変数の設定
    for (const auto& l : lines) {
        std::string current_line = trim(l);
        //行の1番最初に"SET"があるか
        if(current_line.rfind("SET ", 0) == 0) {
            //SETの後の文字列を取得
            std::string content = trim(current_line.substr(4));
            //'='の位置を取得
            std::size_t eq_pos = content.find('=');
            if (eq_pos != std::string::npos) {
                //最初から'='までをkeyに、'='から最後までをvalにセット
                std::string key = trim(content.substr(0, eq_pos));
                std::string val = trim(content.substr(eq_pos + 1));

                variables_[key] = val;
            }
        }
    }

    //コマンドの設定
    for (const auto& l : lines) {
        std::string current_line = trim(l);
        //行の空白、もしくは行の最初が'#'・"SET"の場合はスキップ
        if (current_line.empty() || current_line[0] == '#' || current_line.rfind("SET", 0) == 0) {
            continue;
        }

        Command cmd;
        std::string command_part = current_line;

        //シミュレーション後の構造の保存場所
        std::size_t redirect_pos = command_part.find(">>");
        if (redirect_pos != std::string::npos) {
            command_part = trim(current_line.substr(0, redirect_pos));
            cmd.redirect_target = substitute_vars(trim(current_line.substr(redirect_pos + 2)), variables_);
        }

        //コマンドを空白で分割
        std::vector<std::string> tokens = split(command_part, ' ');

        //コマンド名の取得
        cmd.name = tokens[0];

        //コマンド設定の取得
        for (std::size_t i = 1; i < tokens.size(); i ++) {
            //トークンの最初に"--"がある場合
            if (tokens[i].rfind("--", 0) == 0) {
                std::string arg = tokens[i].substr(2);
                std::size_t eq_pos = arg.find('=');
                std::string key = arg.substr(0, eq_pos);
                //もし値が変数名だった場合、変数をその中身に置き換える
                std::string val = substitute_vars(arg.substr(eq_pos + 1), variables_);
                cmd.args[key] = val;
            }
        }
        commands_.push_back(cmd);
    }
}