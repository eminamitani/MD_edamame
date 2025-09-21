#ifndef THERMOSTAT_HPP
#define THERMOSTAT_HPP

class Thermostat {
    public: 
        virtual ~Thermostat() {}
        
        virtual void update() = 0;
        virtual void set_temp() = 0;
};

#endif