# CST435 Assignment 2: Parallel Image Processing System

## 📌 Project Overview
This project implements a high-performance image processing pipeline capable of applying distinct filters to images from the Food-101 dataset. The system demonstrates parallel computing concepts using:
1. **OpenMP** (Implicit threading)
2. **C++ Standard Threads** (Explicit threading with `std::thread`)

## 👥 Group Members
* **[Name 1]** (Matric No)
* **[Name 2]** (Matric No)
* **[Name 3]** (Matric No)
* **[Name 4]** (Matric No)

## 🛠 System Requirements
* **OS:** Linux (Ubuntu 22.04 LTS) / Windows
* **Compiler:** G++ (GCC) with OpenMP support
* **Language:** C++17

## 📂 Project Structure
```text
CST435_Assignment2/
├── data/              # Image dataset (Local only, ignored by Git)
├── include/           # Header-only libraries (stb_image)
├── src_openmp/        # Source code for OpenMP implementation
├── src_threads/       # Source code for std::thread implementation
└── README.md          # Project documentation
