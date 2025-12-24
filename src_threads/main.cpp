#include <iostream>
#include <vector>
#include <thread> // Thread Header
#include <string>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

namespace fs = std::filesystem;

void grayscaleWorker(unsigned char* img, int w, int h, int c, int startRow, int endRow) {
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * c;
            if (c >= 3) {
                unsigned char gray = (unsigned char)(0.299f*img[idx] + 0.587f*img[idx+1] + 0.114f*img[idx+2]);
                img[idx] = gray; img[idx+1] = gray; img[idx+2] = gray;
            }
        }
    }
}

void applyGrayscaleThreads(unsigned char* img, int w, int h, int c, int numThreads) {
    std::vector<std::thread> workers;
    int rowsPerThread = h / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        int start = t * rowsPerThread;
        int end = (t == numThreads - 1) ? h : start + rowsPerThread;
        workers.emplace_back(grayscaleWorker, img, w, h, c, start, end);
    }
    for (auto& t : workers) t.join();
}

int main() {
    int numThreads = 4;
    std::string inputDir = "../data/images";
    std::string outputDir = "../data/output_threads";
    if (!fs::exists(outputDir)) fs::create_directories(outputDir);

    std::cout << "Starting Thread Processing (" << numThreads << " threads)..." << std::endl;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        std::string path = entry.path().string();
        if (path.find(".jpg") == std::string::npos) continue;

        int w, h, c;
        unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 0);
        if (!img) continue;

        applyGrayscaleThreads(img, w, h, c, numThreads);

        std::string outPath = outputDir + "/" + entry.path().filename().string();
        stbi_write_jpg(outPath.c_str(), w, h, c, img, 90);
        stbi_image_free(img);
    }
    return 0;
}