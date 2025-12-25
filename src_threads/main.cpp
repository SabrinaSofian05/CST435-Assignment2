/**
 * @file main.cpp
 * @brief Full Mode: Loops 1, 2, 4, 8 threads, SAVES IMAGES, and prints Performance Table.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <chrono>
#include <iomanip> // For table formatting

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

namespace fs = std::filesystem;
using namespace std;

// --- 1. PARALLEL ENGINE ---
template<typename Func, typename... Args>
void runParallel(int numThreads, int height, Func f, Args... args) {
    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int s = i * rowsPerThread;
        int e = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        threads.emplace_back(f, args..., s, e);
    }
    for (auto& t : threads) t.join();
}

// --- 2. FILTERS ---
void applyGrayscale(const unsigned char* input, unsigned char* output, int width, int channels, int start, int end) {
    if (channels < 3) return;
    for (int y = start; y < end; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = (y * width + x) * channels;
            unsigned char gray = (unsigned char)(0.299f * input[i] + 0.587f * input[i+1] + 0.114f * input[i+2]);
            output[i] = output[i+1] = output[i+2] = gray;
            if (channels == 4) output[i+3] = input[i+3];
        }
    }
}

void applyConvolution(const unsigned char* input, unsigned char* output, int width, int height, int channels, const float k[3][3], int start, int end) {
    for (int y = start; y < end; ++y) {
        if (y == 0 || y >= height - 1) continue;
        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        sum += input[((y + ky) * width + (x + kx)) * channels + c] * k[ky+1][kx+1];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)max(0.0f, min(255.0f, sum));
            }
        }
    }
}

void applyBlur(const unsigned char* in, unsigned char* out, int w, int h, int c, int s, int e) {
    float k[3][3] = {{1/16.f, 2/16.f, 1/16.f}, {2/16.f, 4/16.f, 2/16.f}, {1/16.f, 2/16.f, 1/16.f}};
    applyConvolution(in, out, w, h, c, k, s, e);
}

void applySharpen(const unsigned char* in, unsigned char* out, int w, int h, int c, int s, int e) {
    float k[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    applyConvolution(in, out, w, h, c, k, s, e);
}

void applyEdge(const unsigned char* input, unsigned char* output, int w, int h, int c, int s, int e) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int y = s; y < e; ++y) {
        if (y == 0 || y >= h - 1) continue;
        for (int x = 1; x < w - 1; ++x) {
            for (int k = 0; k < c; ++k) {
                float sumX = 0, sumY = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int idx = ((y + ky) * w + (x + kx)) * c + k;
                        sumX += input[idx] * gx[ky+1][kx+1];
                        sumY += input[idx] * gy[ky+1][kx+1];
                    }
                }
                output[(y*w+x)*c + k] = (unsigned char)max(0.0f, min(255.0f, sqrt(sumX*sumX + sumY*sumY)));
            }
        }
    }
}

void applyBrightness(const unsigned char* input, unsigned char* output, int w, int h, int c, int val, int s, int e) {
    for (int y = s; y < e; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * c;
            for (int k = 0; k < c; ++k) output[idx+k] = (unsigned char)max(0, min(255, input[idx+k] + val));
        }
    }
}

// --- 3. MAIN LOOP ---
int main() {
    std::string inDir = "images";
    std::vector<int> testThreads = {1, 2, 4, 8}; 
    struct Result { int t; double time; };
    std::vector<Result> results;

    struct Image { std::string name; unsigned char* data; int w, h, c; };
    std::vector<Image> loadedImages;

    // UI Header
    std::cout << "LOADING IMAGES INTO RAM...\n";
    std::cout << "===========================================\n";

    if (!fs::exists(inDir)) {
        std::cout << "ERROR: 'images' folder not found!\n";
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(inDir)) {
        if (entry.path().extension() == ".jpg") {
            int w, h, c;
            unsigned char* d = stbi_load(entry.path().string().c_str(), &w, &h, &c, 0);
            if (d) loadedImages.push_back({entry.path().filename().string(), d, w, h, c});
        }
    }
    std::cout << "Loaded " << loadedImages.size() << " images.\n\n";

    unsigned char* buffer = (unsigned char*)malloc(4000 * 4000 * 4); // Big buffer

    // --- AUTOMATIC TESTING LOOP ---
    for (int n : testThreads) {
        std::cout << "Testing " << n << " Thread(s)... " << std::flush;
        
        // Create unique output folder for this thread count
        std::string outDir = "output_threads_" + std::to_string(n);
        if (!fs::exists(outDir)) fs::create_directories(outDir);

        auto start = std::chrono::high_resolution_clock::now();

        // Process loop
        for (auto& img : loadedImages) {
            runParallel(n, img.h, applyGrayscale, img.data, buffer, img.w, img.c);
            stbi_write_jpg((outDir + "/gray_" + img.name).c_str(), img.w, img.h, img.c, buffer, 80);

            runParallel(n, img.h, applyBlur, img.data, buffer, img.w, img.h, img.c);
            stbi_write_jpg((outDir + "/blur_" + img.name).c_str(), img.w, img.h, img.c, buffer, 80);

            runParallel(n, img.h, applyEdge, img.data, buffer, img.w, img.h, img.c);
            stbi_write_jpg((outDir + "/edge_" + img.name).c_str(), img.w, img.h, img.c, buffer, 80);

            runParallel(n, img.h, applySharpen, img.data, buffer, img.w, img.h, img.c);
            stbi_write_jpg((outDir + "/sharp_" + img.name).c_str(), img.w, img.h, img.c, buffer, 80);

            runParallel(n, img.h, applyBrightness, img.data, buffer, img.w, img.h, img.c, 50);
            stbi_write_jpg((outDir + "/bright_" + img.name).c_str(), img.w, img.h, img.c, buffer, 80);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        results.push_back({n, diff.count()});
        
        std::cout << "Done! (" << diff.count() << "s)\n";
    }

    // --- FINAL OUTPUT ---
    std::cout << "\n===========================================\n";
    std::cout << "   PERFORMANCE SUMMARY (std::thread)\n";
    std::cout << "===========================================\n";
    std::cout << " Threads | Total Time (s) | Speedup \n";
    std::cout << "---------|----------------|---------\n";

    double baseTime = results[0].time;
    for (const auto& r : results) {
        double speedup = baseTime / r.time;
        std::cout << "    " << r.t << "    |    " 
                  << std::fixed << std::setprecision(4) << r.time << "    |  " 
                  << std::setprecision(2) << speedup << "x\n";
    }
    std::cout << "===========================================\n";

    free(buffer);
    for (auto& img : loadedImages) stbi_image_free(img.data);
    return 0;
}
