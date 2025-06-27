#ifndef _TIMER_H_
#define _TIMER_H_


#include <thread>
#include <chrono>
#include "Layer/Layer.h"

class Timer {
public:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> duration;

    Layer* l = nullptr;
    Timer(Layer* l);
    ~Timer();
};

#include "Timer.hpp"

#endif