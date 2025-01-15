# CUDA-Based John Conway’s Game of Life

This project implements **John Conway’s Game of Life** using **C++** and **CUDA** to leverage GPU processing for improved performance. The program supports various types of memory allocation and utilizes dynamic thread configuration for parallel computation.

---

## Objective

To explore GPU parallelism using CUDA and understand the impact of memory allocation techniques (Normal, Pinned, and Managed) on computational efficiency in a dynamic, grid-based simulation.

---

## Features

1. **CUDA Parallelization**:
   - Utilizes CUDA kernels to compute cell states in parallel across the grid.
   - Configurable number of threads per block.

2. **Memory Allocation**:
   - Supports **Normal**, **Pinned**, and **Managed** memory for data storage and transfer between CPU and GPU.

3. **Dynamic Grid Configuration**:
   - Accepts runtime arguments to specify grid dimensions, cell size, and thread configuration.

4. **Randomized Initialization**:
   - Grid cells are randomly initialized as alive or dead.

5. **Real-Time Visualization**:
   - Displays the current state of the grid in a graphical window.
   - Alive cells are displayed in white; dead cells are left undrawn (black background).

6. **Performance Metrics**:
   - Outputs the processing time (in microseconds) for the last 100 generations to the console.

---

## Gameplay Instructions

### Controls:
- **Escape**: Exit the simulation.

### Objective:
- Observe the evolving generations of Conway's Game of Life using GPU-accelerated computation.

### Graphics:
- Cells:
  - **Alive**: White.
  - **Dead**: Black (not drawn).

---

## Command-Line Arguments

The program accepts up to 5 command-line arguments to configure the simulation:

```bash
./Lab4 -n <threads_per_block> -c <cell_size> -x <window_width> -y <window_height> -t <memory_type>
```

## Flags:
- -n: Number of threads per block (must be a multiple of 32, defaults to 32).
- -c: Cell size (square, >=1, defaults to 5).
- -x: Window width (defaults to 800).
- -y: Window height (defaults to 600).
- -t: Memory type (NORMAL, PINNED, or MANAGED, defaults to NORMAL).

## File Structure

### Source Files:
- **cuda_kernels.cu**: Implements CUDA kernels for cell state updates and memory management.
- **main.cpp**: Contains the main logic, command-line argument parsing, and graphical window management.

---

## How to Run

### Prerequisites:
- Ensure your system supports CUDA.
- **C++ Compiler**: GCC with CUDA support.

### Steps:
1. Clone the respository and navigate to the project directory.
2. Compile the code
3. Run the program
   ```bash
   ./Lab4 -n <num_threads> -c <cell_size> -x <window_width> -y <window_height> -t <processing_type>
   ```

## Example Console Output
```bash
100 generations took 450 microsecs with 32 threads per block using Normal memory allocation.
100 generations took 390 microsecs with 64 threads per block using Pinned memory allocation.
100 generations took 320 microsecs with 128 threads per block using Managed memory allocation.
```

## Testing and Debugging

### Key Features to Test:

1. **Command-Line Arguments**:
   - Validate default values when arguments are missing.
   - Test various combinations of valid arguments.

2. **Memory Allocation**:
   - Compare performance for `NORMAL`, `PINNED`, and `MANAGED` memory types.

3. **CUDA Kernels**:
   - Verify the correctness of cell state updates in each generation.

4. **Random Grid Initialization**:
   - Ensure proper randomization of initial cell states.

5. **Performance**:
   - Measure and compare processing times for different thread configurations and memory allocation types.

