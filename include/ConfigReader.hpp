#ifndef CONFIG_READER_HPP
#define CONFIG_READER_HPP

#include <string>
#include <vector>
#include <map>
#include "Command.hpp"

class ConfigReader {
    public:
        ConfigReader(const std::string& path);
        const std::vector<Command>& commands() const { return commands_; }
        const std::map<std::string, std::string>& variables() const { return variables_; }
    
    private:
        void parse(const std::string& filename);

        std::map<std::string, std::string> variables_;
        std::vector<Command> commands_;
};

#endif