﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Helper function to convert compute capability to the number of cores
int ConvertSMVer2Cores(int major, int minor) {
    // Refer to NVIDIA CUDA Programming Guide for the compute capability to cores conversion
    // This is a simplified version and may not cover all cases
    int cores;

    switch ((major << 4) + minor) {
    case 0x10:
        cores = 8;
        break;
    case 0x11:
    case 0x12:
        cores = 8;
        break;
    case 0x13:
        cores = 32;
        break;
    case 0x20:
        cores = 32;
        break;
    default:
        cores = 0;
        break;
    }

    return cores;
}


int main() {
    int device_id = 0; // Device ID (you can change it if you have multiple devices)
    cudaSetDevice(device_id); // Set the device to use

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    int num_SM = device_prop.multiProcessorCount;
    int max_threads_per_SM = device_prop.maxThreadsPerMultiProcessor;
    int warp_size = device_prop.warpSize;

    int threads_scheduled = num_SM * (max_threads_per_SM / warp_size);

    printf("Number of Streaming Multiprocessors: %d\n", num_SM);
    printf("Max Threads per Multiprocessor: %d\n", max_threads_per_SM);
    printf("Warp Size: %d\n", warp_size);
    printf("Threads Scheduled: %d\n", threads_scheduled);

    return 0;
}