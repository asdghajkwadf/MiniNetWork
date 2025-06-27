#include <thread>
#include <chrono>
#include "Layer/Layer.h"
#include "Timer.h"

Timer::Timer(Layer* l)
{
    start = std::chrono::high_resolution_clock::now();
    l->callTimes += 1;
    this->l = l;
}

Timer::~Timer()
{
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    float ms = duration.count() * 1000.0f;
    // std::std::cout << "Timer took " << ms << "ms" << std::endl;
    l->COST_TIME += ms;
}