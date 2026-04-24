#pragma once

#include <cassert>

#include "check.h"

/**
 * Simple GPU Timer using cuda events.
 */
class gpuTimer {

private:

	cudaEvent_t _start, _stop;

	float _gpuTime;

public:

	gpuTimer() : _start(nullptr), _stop(nullptr), _gpuTime(0) 
	{
		// Create events.
		// An event is defined as a point in time on a stream.
		if (_start == nullptr) checkErrors(cudaEventCreate(&_start), "Creating start event");
		if (_stop == nullptr) checkErrors(cudaEventCreate(&_stop), "Creating stop event");
	}

	~gpuTimer() {
		// Destroy events. Since they are pointers, 
		// destroying them ensures their memory is freed.
		if (_start != nullptr) checkErrors(cudaEventDestroy(_start), "Destroying start event");
		if (_stop != nullptr) checkErrors(cudaEventDestroy(_stop), "Destroying stop event");

		_start = nullptr, _stop = nullptr;
	}

	inline void  start  (const cudaStream_t& _s = 0) { 
		assert(_start && _stop);
		// Record the start event on the given stream.
		checkErrors(cudaEventRecord(_start, _s), "Recording start event"); 
	}

	inline void  stop   (const cudaStream_t& _s = 0) { 
		assert(_start && _stop);
		// Record the stop event on the given stream.
		checkErrors(cudaEventRecord(_stop, _s), "Recording stop event"); 
	}
	
	inline float elapsed() {
		assert(_start && _stop);
		_gpuTime = 0;
		// Wait for the stop event to complete.
		checkErrors(cudaEventSynchronize(_stop), "Synchronizing stop event");
		// Calculate the elapsed time between the two events.
		checkErrors(cudaEventElapsedTime(&_gpuTime, _start, _stop), "Calculating elapsed time");
		// Return the elapsed time in milliseconds.
		return _gpuTime;
	}
};