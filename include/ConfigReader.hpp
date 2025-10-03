/**
* @file ConfigReader.hpp
* @brief 設定ファイルを読み込む
*/

#ifndef CONFIG_READER_HPP
#define CONFIG_READER_HPP

#include <string>
#include <vector>
#include <map>
#include "Command.hpp"

class ConfigReader {
    public:
        ConfigReader(const std::string& path);
        /**
         * @brief すべてのコマンドを取得
         * @return コマンド
         */
        const std::vector<Command>& commands() const { return commands_; }
        /**
         * @brief 設定ファイルで設定したすべての変数を取得
         * @return 変数名・値のマップ
         */
        const std::map<std::string, std::string>& variables() const { return variables_; }
    
    private:
        void parse(const std::string& filename);

        std::map<std::string, std::string> variables_;
        std::vector<Command> commands_;
};

#endif