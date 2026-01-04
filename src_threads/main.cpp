/**
 * @file main.cpp
 * @brief Parallel Image Processing using std::thread - Sequential Pipeline
 * @course CST435: Parallel Computing
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <chrono>

// STB Image Libraries
// Ensure stb_image.h and stb_image_write.h are in the ../include/ folder
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

namespace fs = std::filesystem;
using namespace std;

// ==========================================
//             PARALLEL HELPER 
// ==========================================
template<typename Func, typename... Args>
void runParallel(int numThreads, int height, Func f, Args... args) {
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        // Ensure the last thread covers any remaining rows
        int endRow = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        
        threads.emplace_back(f, args..., startRow, endRow);
    }

    for (auto& t : threads) t.join();
}

// ==========================================
// IMAGE FILTER FUNCTIONS
// ==========================================

// 1. Grayscale
void applyGrayscale(const unsigned char* input, unsigned char* output, int width, int channels, int startRow, int endRow) {
    if (channels < 3) return; 
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = (y * width + x) * channels;
            unsigned char gray = (unsigned char)(0.299f * input[i] + 0.587f * input[i + 1] + 0.114f * input[i + 2]);
            
            output[i] = gray;
            if (channels >= 3) { output[i+1] = gray; output[i+2] = gray; }
            if (channels == 4) output[i+3] = input[i+3];
        }
    }
}

// Convolution Helper
void applyConvolution(const unsigned char* input, unsigned char* output, int width, int height, int channels, const float kernel[3][3], int startRow, int endRow) {
    for (int y = startRow; y < endRow; ++y) {
        if (y == 0 || y >= height - 1) continue; 

        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pixelIdx = ((y + ky) * width + (x + kx)) * channels + c;
                        sum += input[pixelIdx] * kernel[ky + 1][kx + 1];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)max(0.0f, min(255.0f, sum));
            }
        }
    }
}

// 2. Blur
void applyBlur(const unsigned char* input, unsigned char* output, int width, int height, int channels, int startRow, int endRow) {
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };
    applyConvolution(input, output, width, height, channels, kernel, startRow, endRow);
}

// 3. Sharpen
void applySharpen(const unsigned char* input, unsigned char* output, int width, int height, int channels, int startRow, int endRow) {
    float kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    applyConvolution(input, output, width, height, channels, kernel, startRow, endRow);
}

// 4. Edge (Sobel)
void applyEdge(const unsigned char* input, unsigned char* output, int width, int height, int channels, int startRow, int endRow) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = startRow; y < endRow; ++y) {
        if (y == 0 || y >= height - 1) continue;

        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sumX = 0.0f, sumY = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pixelIdx = ((y + ky) * width + (x + kx)) * channels + c;
                        sumX += input[pixelIdx] * gx[ky + 1][kx + 1];
                        sumY += input[pixelIdx] * gy[ky + 1][kx + 1];
                    }
                }
                int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
                output[(y * width + x) * channels + c] = (unsigned char)max(0, min(255, magnitude));
            }
        }
    }
}

// 5. Brightness
void applyBrightness(const unsigned char* input, unsigned char* output, int width, int height, int channels, int value, int startRow, int endRow) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                if (channels == 4 && c == 3) { 
                     output[idx+c] = input[idx+c];
                     continue;
                }
                int newVal = input[idx + c] + value;
                output[idx + c] = (unsigned char)max(0, min(255, newVal));
            }
        }
    }
}

// ==========================================
// MAIN BATCH PROCESSOR
// ==========================================
int main(int argc, char* argv[]) {
    // 1. Read Thread Count from Command Line
    int numThreads = (argc > 1) ? atoi(argv[1]) : 4;

    std::string inputFolder = "../data/images"; 
    // Create output folder for combined results
    std::string outputFolder = "../output/combined_" + std::to_string(numThreads) + "threads"; 
    
    if (!fs::exists(outputFolder)) fs::create_directories(outputFolder);

    // --- UI HEADER  ---
    std::cout << "===========================================" << std::endl;
    std::cout << "   STARTING PIPELINE PROCESSOR (" << numThreads << " Threads)" << std::endl;
    std::cout << "===========================================" << std::endl;

    if (!fs::exists(inputFolder)) {
        std::cout << "Error: Input folder '" << inputFolder << "' not found." << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int fileCount = 0;

    // --- PIPELINE BUFFERS ---
    // Buffer A and Buffer B allow us to swap input/output between steps without race conditions
    // 4000x4000x4 is a safe size for most standard images; adjust if processing 4K/8K images.
    unsigned char* bufferA = (unsigned char*)malloc(4000 * 4000 * 4);
    unsigned char* bufferB = (unsigned char*)malloc(4000 * 4000 * 4); 

    if (!bufferA || !bufferB) {
        std::cout << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // BATCH LOOP
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();
        
        // Simple extension check
        if (path.find(".jpg") == std::string::npos && 
            path.find(".jpeg") == std::string::npos && 
            path.find(".png") == std::string::npos) continue;

        std::cout << "Processing: " << filename << " ... " << std::flush;

        int width, height, channels;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!img) { std::cout << "Failed to load!" << std::endl; continue; }

        fileCount++;

        // --- PIPELINE EXECUTION ---
        // Logic: Input -> BufferA -> BufferB -> BufferA ... -> Final Save

        // Step 1: Grayscale (Input: Original img -> Output: bufferA)
        runParallel(numThreads, height, applyGrayscale, img, bufferA, width, channels);

        // Step 2: Blur (Input: bufferA -> Output: bufferB)
        runParallel(numThreads, height, applyBlur, bufferA, bufferB, width, height, channels);

        // Step 3: Sharpen (Input: bufferB -> Output: bufferA)
        runParallel(numThreads, height, applySharpen, bufferB, bufferA, width, height, channels);

        // Step 4: Edge/Sobel (Input: bufferA -> Output: bufferB)
        runParallel(numThreads, height, applyEdge, bufferA, bufferB, width, height, channels);

        // Step 5: Brightness (Input: bufferB -> Output: bufferA)
        // Note: The final result ends up in BufferA
        runParallel(numThreads, height, applyBrightness, bufferB, bufferA, width, height, channels, 50);

        // --- SAVE FINAL RESULT ---
        // Save the content of bufferA (which holds the result of step 5)
        std::string savePath = outputFolder + "/final_" + filename;
        stbi_write_jpg(savePath.c_str(), width, height, channels, bufferA, 90);

        stbi_image_free(img);
        std::cout << "Done." << std::endl;
    }

    free(bufferA);
    free(bufferB);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // --- FINAL STATS ---
    std::cout << "\n===========================================" << std::endl;
    std::cout << "   COMPLETED!" << std::endl;
    std::cout << "   Images Processed: " << fileCount << std::endl;
    std::cout << "   Threads Used:     " << numThreads << std::endl;
    std::cout << "   TOTAL TIME:       " << diff.count() << " seconds" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    return 0;
}
