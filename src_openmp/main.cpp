#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <omp.h> // OpenMP Header

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

namespace fs = std::filesystem;

void applyGrayscale(unsigned char* img, int width, int height, int channels) {
    int size = width * height * channels;
    // OpenMP Parallel Loop
    #pragma omp parallel for
    for (int i = 0; i < size; i += channels) {
        if (channels >= 3) {
            unsigned char r = img[i];
            unsigned char g = img[i+1];
            unsigned char b = img[i+2];
            unsigned char gray = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
            img[i] = gray;
            img[i+1] = gray;
            img[i+2] = gray;
        }
    }
}

int main() {
    std::string inputDir = "../data/images";
    std::string outputDir = "../data/output_openmp";
    if (!fs::exists(outputDir)) fs::create_directories(outputDir);

    std::cout << "Starting OpenMP Processing..." << std::endl;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        std::string path = entry.path().string();
        if (path.find(".jpg") == std::string::npos) continue;

        int w, h, c;
        unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 0);
        if (!img) continue;

        applyGrayscale(img, w, h, c);

        std::string outPath = outputDir + "/" + entry.path().filename().string();
        stbi_write_jpg(outPath.c_str(), w, h, c, img, 90);
        stbi_image_free(img);
        std::cout << "Processed: " << entry.path().filename().string() << std::endl;
    }
    return 0;
}