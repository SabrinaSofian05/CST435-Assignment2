/**
 * @file main.cpp
 * @brief Parallel Image Processing using OpenMP
 * @course CST435: Parallel Computing
 * * Objectives addressed:
 * 1. Data decomposition using OpenMP 'parallel for'
 * 2. Performance optimization via loop collapsing and memory reuse
 * 3. Benchmarking across varying thread counts on GCP
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <filesystem>
#include <chrono>

// STB Image Libraries for loading and saving images
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

namespace fs = std::filesystem;
using namespace std;

// ==========================================
// PARALLEL IMAGE FILTER FUNCTIONS (OpenMP)
// ==========================================

// 1. Grayscale Conversion: RGB -> Gray
// Apply Luminance formula: Y = 0.299R + 0.587G + 0.114B
void applyGrayscale(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    if (channels < 3) return; 

    // Parallelize the loop: Each thread handles a chunk of pixels
    #pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        int r = input[i * channels];
        int g = input[i * channels + 1];
        int b = input[i * channels + 2];
        // Luminance formula
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        output[i * channels] = gray;
        output[i * channels + 1] = gray;
        output[i * channels + 2] = gray;
        if (channels == 4) output[i * channels + 3] = input[i * channels + 3];
    }
}

// Helper for Convolution (Used by Blur, Sharpen, Edge)
void applyConvolution(unsigned char* input, unsigned char* output, int width, int height, int channels, const float kernel[3][3]) {
    // collapse(2) tells OpenMP to treat the 2D grid as one giant list of tasks
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pixelIdx = ((y + ky) * width + (x + kx)) * channels + c;
                        sum += input[pixelIdx] * kernel[ky + 1][kx + 1];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)max(0, min(255, (int)sum));
            }
        }
    }
}

// 2. Gaussian Blur (3x3 Kernel)
void applyBlur(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };
    applyConvolution(input, output, width, height, channels, kernel);
}

// 3. Sharpening (3x3 Kernel)
void applySharpen(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    float kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    applyConvolution(input, output, width, height, channels, kernel);
}

// 4. Edge Detection (Sobel Operator)
void applyEdge(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Sobel requires separate X and Y kernels
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; ++y) {
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
                // Calculate magnitude of edge
                int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
                output[(y * width + x) * channels + c] = (unsigned char)max(0, min(255, magnitude));
            }
        }
    }
}

// 5. Brightness Adjustment
void applyBrightness(unsigned char* input, unsigned char* output, int width, int height, int channels, int value) {
    #pragma omp parallel for
    for (int i = 0; i < width * height * channels; ++i) {
        int newVal = input[i] + value;
        output[i] = (unsigned char)max(0, min(255, newVal));
    }
}

// ==========================================
// MAIN BATCH PIPELINE PROCESSOR
// ==========================================
int main(int argc, char* argv[]) {
    std::string inputFolder = "../data/images"; // input folder
    std::string outputFolder = "../output/openmp";  // output folder
    
    // THREAD SETUP: Allows testing scalability (1, 2, 4, 8 threads)
    int numThreads = (argc > 1) ? atoi(argv[1]) : 4;
    omp_set_num_threads(numThreads);

    // Create output folder
    if (!fs::exists(outputFolder)) fs::create_directories(outputFolder);

    // --- UI HEADER ---
    std::cout << "===========================================" << std::endl;
    std::cout << "   STARTING BATCH PROCESSOR (" << numThreads << " Threads)" << std::endl;
    std::cout << "   [OpenMP Implementation]" << std::endl;
    std::cout << "===========================================" << std::endl;

    if (!fs::exists(inputFolder)) {
        std::cout << "Error: Input folder '" << inputFolder << "' not found." << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int fileCount = 0;

    // Allocae two large buffers to swap between them (Supports up to 4K resolution images)
    size_t bufferSize = 4000 * 4000 * 4; // width * height * max channels
    unsigned char* bufferA = (unsigned char*)malloc(bufferSize);
    unsigned char* bufferB = (unsigned char*)malloc(bufferSize);

    if (!bufferA || !bufferB) {
        std::cout << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // BATCH LOOP: Processes each dataset sequentially
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();
        
        // Skip non-images
        if (path.find(".jpg") == std::string::npos && path.find(".jpeg") == std::string::npos && path.find(".png") == std::string::npos) continue;

        // Separate base name and extension for better naming
        size_t lastDot = filename.find_last_of(".");
        string baseName = filename.substr(0, lastDot); 
        string ext = filename.substr(lastDot);

        // --- PRINT PROGRESS ---
        std::cout << "Processing: " << filename << " ... ";

        int width, height, channels;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!img) { std::cout << "Failed to load!" << std::endl; continue; }

        // --- THE PIPELINE SEQUENCE ---

        // 1. Grayscale (image -> buffer A)
        applyGrayscale(img, bufferA, width, height, channels);
        //stbi_write_jpg((outputFolder + "/" + baseName + "_grayscale" + ext).c_str(), width, height, 1, grayImg, 100);

        // 2. Blur (buffer A -> buffer B)
        applyBlur(bufferA, bufferB, width, height, channels);
        //stbi_write_jpg((outputFolder + "/" + baseName + "_blur" + ext).c_str(), width, height, channels, outputImg, 100);

        // 3. Edge (Sobel) (buffer B -> buffer A)
        applyEdge(bufferB, bufferA, width, height, channels);
        //stbi_write_jpg((outputFolder + "/" + baseName + "_edge" + ext).c_str(), width, height, channels, outputImg, 100);

        // 4. Sharpen (buffer A -> buffer B)
        applySharpen(bufferA, bufferB, width, height, channels);
        //stbi_write_jpg((outputFolder + "/" + baseName + "_sharpen" + ext).c_str(), width, height, channels, outputImg, 100);

        // 5. Brightness (buffer B -> buffer A)
        applyBrightness(bufferB, bufferA, width, height, channels, 50);
        //stbi_write_jpg((outputFolder + "/" + baseName + "_bright" + ext).c_str(), width, height, channels, outputImg, 100);

        // Save final result from the last active buffer (bufferA)
        string outPath = outputFolder + "/" + baseName + "_output.jpg";
        //stbi_write_jpg(outPath.c_str(), width, height, channels, bufferA, 100);

        // Cleanup
        stbi_image_free(img);
        fileCount++; 
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