# CST435 Assignment 2: Parallel Image Processing System

## 👥 Team Information
* **Course:** CST435 Parallel and Distributed Computing
* **Group:** Group 46
* **Members:**
  1. [Name 1] ([Matric No])
  2. [Name 2] ([Matric No])
  3. [Name 3] ([Matric No])
  4. [Name 4] ([Matric No])

---

## 📌 Project Overview
This system implements a high-performance image processing pipeline that applies five filters (Grayscale, Blur, Edge Detection, Sharpen, and Brightness) to the Food-101 dataset. The project demonstrates parallel computing scalability by comparing:
* **OpenMP** (Implicit threading using compiler directives)
* **C++ Standard Threads** (Explicit threading using `std::thread`)

---

## Execution Guide
### Option 1: Benchmark File
The Benchmark Manager automatically compiles both implementations and runs them across 1, 2, 4, and 8 threads to generate a live comparison table.

```bash
cd benchmark
g++ benchmark.cpp -o manager
./manager 
```

### Option 2: Manual Execution (in folder openmp / threads)
#### OpenMP Implementation: 
```bash
cd src_openmp
g++ main.cpp -o main -fopenmp -std=c++17 -I../include
./main 4 #(4 threads)
```
#### C++ Threads Implementation:
```bash
cd src_threads
g++ main.cpp -o main -fopenmp -std=c++17 -I../include
./main 4 #(4 threads)
```
---

## 📂 Project Structure
```text
CST435_Assignment2/
├── benchmark/         # C++ Benchmark Manager to automate tests
├── data/              # Input images (data/images)
├── include/           # Header-only libraries (stb_image.h)
├── output/            # Processed images (openmp/ and threads/)
├── src_openmp/        # Source code for OpenMP implementation
├── src_threads/       # Source code for std::thread implementation
└── README.md          # Project documentation
