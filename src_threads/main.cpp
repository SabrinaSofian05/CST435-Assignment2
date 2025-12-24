#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <chrono> // For the Timer

namespace fs = std::filesystem;

// IMPLEMENTATION (STB)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

// --- FILTER LOGIC ---

// 1. Grayscale
void filterGrayscale(unsigned char* img, int width, int channels, int startRow, int endRow) {
    if (channels < 3) return;
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            unsigned char r = img[idx];
            unsigned char g = img[idx + 1];
            unsigned char b = img[idx + 2];
            unsigned char gray = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
            img[idx] = gray; img[idx+1] = gray; img[idx+2] = gray;
        }
    }
}

// 2. Brightness
void filterBrightness(const unsigned char* input, unsigned char* output, int width, int channels, int startRow, int endRow, int value) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            for(int c = 0; c < channels; c++) { 
                if (c == 3) { output[idx + c] = input[idx + c]; continue; }
                int p = input[idx + c] + value;
                output[idx + c] = (unsigned char)std::min(std::max(p, 0), 255);
            }
        }
    }
}

// Convolution Helper
void applyKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, 
                 int startRow, int endRow, const float* kernel, int kSize) {
    int pad = kSize / 2;
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y < pad || y >= height - pad || x < pad || x >= width - pad) {
                int idx = (y * width + x) * channels;
                for (int c = 0; c < channels; ++c) output[idx + c] = input[idx + c];
                continue;
            }
            for (int c = 0; c < 3; ++c) { 
                float sum = 0.0f;
                for (int ky = -pad; ky <= pad; ++ky) {
                    for (int kx = -pad; kx <= pad; ++kx) {
                        int pIdx = ((y + ky) * width + (x + kx)) * channels + c;
                        sum += input[pIdx] * kernel[(ky + pad) * kSize + (kx + pad)];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)std::min(std::max(sum, 0.0f), 255.0f);
            }
            if (channels == 4) output[(y * width + x) * channels + 3] = input[(y * width + x) * channels + 3];
        }
    }
}

// 3. Blur
void filterBlur(const unsigned char* input, unsigned char* output, int width, int height, int channels, int startRow, int endRow) {
    float kernel[9] = { 1/16.f, 2/16.f, 1/16.f, 2/16.f, 4/16.f, 2/16.f, 1/16.f, 2/16.f, 1/16.f };
    applyKernel(input, output, width, height, channels, startRow, endRow, kernel, 3);
}

// 4. Sharpen
void filterSharpen(const unsigned char* input, unsigned char* output, int width, int height, int channels, int startRow, int endRow) {
    float kernel[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    applyKernel(input, output, width, height, channels, startRow, endRow, kernel, 3);
}

// 5. Edge
void filterEdge(const unsigned char* input, unsigned char* output, int width, int height, int channels, int startRow, int endRow) {
    float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int pad = 1;
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y < pad || y >= height - pad || x < pad || x >= width - pad) continue;
            for (int c = 0; c < 3; ++c) {
                float sumX = 0.0f, sumY = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pIdx = ((y + ky) * width + (x + kx)) * channels + c;
                        sumX += input[pIdx] * Gx[(ky + 1) * 3 + (kx + 1)];
                        sumY += input[pIdx] * Gy[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                float mag = std::sqrt(sumX * sumX + sumY * sumY);
                output[(y * width + x) * channels + c] = (unsigned char)std::min(std::max(mag, 0.0f), 255.0f);
            }
            if (channels == 4) output[(y * width + x) * channels + 3] = 255;
        }
    }
}

template<typename Func, typename... Args>
void runParallel(int numThreads, int height, Func f, Args... args) {
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * rowsPerThread;
        int end = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        threads.emplace_back(f, args..., start, end);
    }
    for (auto& t : threads) t.join();
}

// --- MAIN ---
int main() {
    std::string inputFolder = "../images";
    std::string outputFolder = "../output";
    
    // --- !!! CHANGE THIS NUMBER FOR YOUR EXPERIMENTS !!! ---
    int numThreads = 8;  // Try 1, 2, 4, 8
    
    if (!fs::exists(outputFolder)) fs::create_directory(outputFolder);
    std::cout << "===========================================" << std::endl;
    std::cout << "   STARTING BATCH PROCESSOR (" << numThreads << " Threads)" << std::endl;
    std::cout << "===========================================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int fileCount = 0;

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();
        
        // Skip non-images
        if (path.find(".jpg") == std::string::npos && path.find(".jpeg") == std::string::npos && path.find(".png") == std::string::npos) continue;

        // --- PRINT PROGRESS (The "List" you wanted) ---
        std::cout << "Processing: " << filename << " ... ";

        int width, height, channels;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!img) { std::cout << "Failed to load!" << std::endl; continue; }

        fileCount++;
        size_t imgSize = width * height * channels;
        unsigned char* outputImg = (unsigned char*)malloc(imgSize);
        unsigned char* grayImg = (unsigned char*)malloc(imgSize); // Copy for grayscale

        // 1. Grayscale
        memcpy(grayImg, img, imgSize);
        runParallel(numThreads, height, filterGrayscale, grayImg, width, channels);
        stbi_write_jpg((outputFolder + "/gray_" + filename).c_str(), width, height, channels, grayImg, 100);

        // 2. Blur
        runParallel(numThreads, height, filterBlur, img, outputImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/blur_" + filename).c_str(), width, height, channels, outputImg, 100);

        // 3. Edge
        runParallel(numThreads, height, filterEdge, img, outputImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/edge_" + filename).c_str(), width, height, channels, outputImg, 100);

        // 4. Sharpen
        runParallel(numThreads, height, filterSharpen, img, outputImg, width, height, channels);
        stbi_write_jpg((outputFolder + "/sharp_" + filename).c_str(), width, height, channels, outputImg, 100);

        // 5. Brightness
        runParallel(numThreads, height, filterBrightness, img, outputImg, width, channels, 50);
        stbi_write_jpg((outputFolder + "/bright_" + filename).c_str(), width, height, channels, outputImg, 100);

        stbi_image_free(img);
        free(outputImg);
        free(grayImg);
        
        std::cout << "Done (5 filters saved)." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "\n===========================================" << std::endl;
    std::cout << "   COMPLETED!" << std::endl;
    std::cout << "   Images Processed: " << fileCount << std::endl;
    std::cout << "   Threads Used:     " << numThreads << std::endl;
    std::cout << "   TOTAL TIME:       " << diff.count() << " seconds" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    return 0;
}