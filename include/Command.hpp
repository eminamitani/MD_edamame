#ifndef COMMAND_HPP
#define COMMAND_HPP

#include <string>
#include <vector>
#include <map>

struct Command {
    std::string name;
    std::map<std::string, std::string> args;
    std::string redirect_target;
};

#endif