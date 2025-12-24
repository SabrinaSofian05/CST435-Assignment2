#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <filesystem>
#include <chrono>

// LIBRARY SETUP
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;
using namespace std;

// ==========================================
// FILTER FUNCTIONS (OpenMP Enabled)
// ==========================================

// 1. Grayscale: RGB -> Gray
void applyGrayscale(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    if (channels < 3) return; 
    #pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        int r = input[i * channels];
        int g = input[i * channels + 1];
        int b = input[i * channels + 2];
        output[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Helper for Convolution (Used by Blur, Sharpen, Edge)
void applyConvolution(unsigned char* input, unsigned char* output, int width, int height, int channels, const float kernel[3][3]) {
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
                int val = (int)sum;
                output[(y * width + x) * channels + c] = (unsigned char)max(0, min(255, val));
            }
        }
    }
}

// 2. Gaussian Blur
void applyBlur(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };
    applyConvolution(input, output, width, height, channels, kernel);
}

// 3. Sharpening
void applySharpen(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    float kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    applyConvolution(input, output, width, height, channels, kernel);
}

// 4. Edge Detection (Sobel)
void applyEdge(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Sobel requires separate X and Y kernels
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sumX = 0.0f;
                float sumY = 0.0f;
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
void applyBrightness(unsigned char* input, unsigned char* output, int width, int height, int channels, int value) {
    #pragma omp parallel for
    for (int i = 0; i < width * height * channels; ++i) {
        int newVal = input[i] + value;
        output[i] = (unsigned char)max(0, min(255, newVal));
    }
}

// ==========================================
// MAIN BATCH PROCESSOR
// ==========================================
int main(int argc, char* argv[]) {
    std::string inputFolder = "../data/images";        // Make sure this folder has your 100 images
    std::string outputFolder = "../output/openmp";
    
    // Default threads = 4, or pass via CLI (e.g., ./main 8)
    int numThreads = (argc > 1) ? atoi(argv[1]) : 4;
    omp_set_num_threads(numThreads);

    // Create output folder
    if (!fs::exists(outputFolder)) fs::create_directories(outputFolder);

    // --- UI HEADER ---
    std::cout << "===========================================" << std::endl;
    std::cout << "   STARTING BATCH PROCESSOR (" << numThreads << " Threads)" << std::endl;
    std::cout << "   [OpenMP Implementation]" << std::endl;
    std::cout << "===========================================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int fileCount = 0;

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();
        
        // Skip non-images
        if (path.find(".jpg") == std::string::npos && path.find(".jpeg") == std::string::npos && path.find(".png") == std::string::npos) continue;

        // --- PRINT PROGRESS ---
        std::cout << "Processing: " << filename << " ... ";

        int width, height, channels;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!img) { std::cout << "Failed to load!" << std::endl; continue; }

        fileCount++;
        size_t imgSize = width * height * channels;
        size_t graySize = width * height; // 1 channel

        // Allocate buffers
        unsigned char* outputImg = (unsigned char*)malloc(imgSize);
        unsigned char* grayImg = (unsigned char*)malloc(graySize);

        // 1. Grayscale (Save as 1 channel JPG)
        applyGrayscale(img, grayImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/gray_" + filename).c_str(), width, height, 1, grayImg, 100);

        // 2. Blur
        applyBlur(img, outputImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/blur_" + filename).c_str(), width, height, channels, outputImg, 100);

        // 3. Edge (Sobel)
        applyEdge(img, outputImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/edge_" + filename).c_str(), width, height, channels, outputImg, 100);

        // 4. Sharpen
        applySharpen(img, outputImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/sharp_" + filename).c_str(), width, height, channels, outputImg, 100);

        // 5. Brightness
        applyBrightness(img, outputImg, width, height, channels, 50);
        stbi_write_jpg((outputFolder + "/bright_" + filename).c_str(), width, height, channels, outputImg, 100);

        // Cleanup
        stbi_image_free(img);
        free(outputImg);
        free(grayImg);
        
        std::cout << "Done." << std::endl;
    }

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