/**
 * Leandro Alan Kim
 * Class: ECE 4122 A
 * Last Modified: 8 Nov 2024
 * 
 * Implement a C++ CUDA program to run the Game of Life.
 */

#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace sf;
using namespace std;

// default window
int WINDOW_WIDTH = 800; // default width
int WINDOW_HEIGHT = 600; // default height
int PIXEL_SIZE = 5; // default cell size
int threadsPerBlock = 32; // default threads/block
string MEMORY_TYPE = "NORMAL"; // default memory type

// calculating grid
inline int GRID_WIDTH()
{
    return WINDOW_WIDTH / PIXEL_SIZE;
}
inline int GRID_HEIGHT()
{
    return WINDOW_HEIGHT / PIXEL_SIZE;
}

/**
 * count neighbors that are alive
 * @param grid current grid
 * @param x x coordinate
 * @param y y coordinate
 * @param width grid width
 * @param height grid height
 * @return number of cells that are alive neighboring
 */
__device__ int countNeighbors(unsigned char* grid, int x, int y, int width, int height)
{
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue; // not counting itself
            // neighbor coordinates
            int nx = (x + i + width) % width;
            int ny = (y + j + height) % height;
            count += grid[ny * width + nx];
        }
    }
    return count;
}

/**
 * update grid
 * @param currentGrid current grid
 * @param nextGrid next grid
 * @param width grid width
 * @param height grid height
 */
__global__ void updateGridKernel(unsigned char* currentGrid, unsigned char* nextGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int neighbors = countNeighbors(currentGrid, x, y, width, height);
    
    if (currentGrid[idx]) {
        nextGrid[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    } else {
        nextGrid[idx] = (neighbors == 3) ? 1 : 0;
    }
}

/**
 * random number generator to set alive and dead
 * @param grid grid to start
 * @param width grid width
 * @param height grid height
 * @param seed random
 */
__global__ void initGridKernel(unsigned char* grid, int width, int height, unsigned long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    curandState state;
    curand_init(seed + y * width + x, 0, 0, &state);
    grid[y * width + x] = (curand_uniform(&state) < 0.5f) ? 1 : 0; // 50% to be alive
}

/**
 * normal memory- pageable
 * @param h_grid host grid pointer
 * @param d_currentGrid device current grid pointer
 * @param d_nextGrid device next grid pointer
 * @param byteSize size of memory
 */
void allocateNormalMemory(unsigned char*& h_grid, unsigned char*& d_currentGrid, unsigned char*& d_nextGrid, size_t byteSize)
{
    h_grid = new unsigned char[byteSize];
    memset(h_grid, 0, byteSize);
    cudaMalloc(&d_currentGrid, byteSize);
    cudaMalloc(&d_nextGrid, byteSize);
    cudaMemset(d_currentGrid, 0, byteSize);
    cudaMemset(d_nextGrid, 0, byteSize);
}

/**
 * pinned memory
 * @param h_grid host grid pointer
 * @param d_currentGrid device current grid pointer
 * @param d_nextGrid device next grid pointer
 * @param byteSize size of memory
 */
void allocatePinnedMemory(unsigned char*& h_grid, unsigned char*& d_currentGrid, unsigned char*& d_nextGrid, size_t byteSize)
{
    cudaMallocHost(&h_grid, byteSize);  // Pinned memory allocation
    memset(h_grid, 0, byteSize);
    cudaMalloc(&d_currentGrid, byteSize);
    cudaMalloc(&d_nextGrid, byteSize);
    cudaMemset(d_currentGrid, 0, byteSize);
    cudaMemset(d_nextGrid, 0, byteSize);
}

/**
 * managed memory - unified memory
 * @param h_grid host grid pointer
 * @param d_currentGrid device current grid pointer
 * @param d_nextGrid device next grid pointer
 * @param byteSize size of memory
 */
void allocateManagedMemory(unsigned char*& h_grid, unsigned char*& d_currentGrid, unsigned char*& d_nextGrid, size_t byteSize)
{
    cudaMallocManaged(&d_currentGrid, byteSize);
    cudaMallocManaged(&d_nextGrid, byteSize);
    cudaMemset(d_currentGrid, 0, byteSize);
    cudaMemset(d_nextGrid, 0, byteSize);
    h_grid = d_currentGrid;  // pointer shared between cpu and gpu
}

/**
 * command line arg
 * @param argc number of arguments
 * @param argv array of argument strings
 */
void arg(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if (arg == "-n")
        {
            threadsPerBlock = stoi(argv[++i]);
            // if not multiple of 32 then default
            if (threadsPerBlock % 32 != 0)
            {
                threadsPerBlock = 32;
            }
        } else if (arg == "-c")
        {
            PIXEL_SIZE = stoi(argv[++i]);
            // cell size has to be min 1 or default
            if (PIXEL_SIZE < 1)
            {
                PIXEL_SIZE = 5;
            }
        } else if (arg == "-x")
        {
            WINDOW_WIDTH = stoi(argv[++i]);
        } else if (arg == "-y" && i + 1 < argc)
        {
            WINDOW_HEIGHT = stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc)
        {
            MEMORY_TYPE = argv[++i];
            if (MEMORY_TYPE != "NORMAL" && MEMORY_TYPE != "PINNED" && MEMORY_TYPE != "MANAGED")
            {
                MEMORY_TYPE = "NORMAL"; // default
            }
        }
    }
}

int main(int argc, char* argv[])
{
    arg(argc, argv);

    // window
    RenderWindow window(VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "CUDA Game of Life");
    window.setFramerateLimit(60);

    // setup memory
    const int gridSize = GRID_WIDTH() * GRID_HEIGHT();
    unsigned char *d_currentGrid, *d_nextGrid, *h_grid;
    size_t byteSize = gridSize * sizeof(unsigned char);

    // allocate memory based on memory type taken in
    if (MEMORY_TYPE == "NORMAL")
    {
        allocateNormalMemory(h_grid, d_currentGrid, d_nextGrid, byteSize);
    } else if (MEMORY_TYPE == "PINNED")
    {
        allocatePinnedMemory(h_grid, d_currentGrid, d_nextGrid, byteSize);
    } else // managed
    {
        allocateManagedMemory(h_grid, d_currentGrid, d_nextGrid, byteSize);
    }

    // config cuda
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 numBlocks((GRID_WIDTH() + blockSize.x - 1) / blockSize.x, (GRID_HEIGHT() + blockSize.y - 1) / blockSize.y);

    // grid
    unsigned long seed = chrono::system_clock::now().time_since_epoch().count();
    initGridKernel<<<numBlocks, blockSize>>>(d_currentGrid, GRID_WIDTH(), GRID_HEIGHT(), seed);
    cudaDeviceSynchronize();

    // copy for normal and pinned memory
    if (MEMORY_TYPE != "MANAGED")
    {
        cudaMemcpy(h_grid, d_currentGrid, byteSize, cudaMemcpyDeviceToHost);
    }

    // track
    using clock = chrono::high_resolution_clock;
    unsigned long numGenerations = 0;
    long long totalTime = 0;

    // main loop
    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed || (event.type == Event::KeyPressed && event.key.code == Keyboard::Escape))
            {
                window.close();
            }
        }

        // start time for curr generation
        auto startTime = clock::now();

        // update grid for memory tyype
        if (MEMORY_TYPE == "MANAGED")
        {
            updateGridKernel<<<numBlocks, blockSize>>>(d_currentGrid, d_nextGrid, GRID_WIDTH(), GRID_HEIGHT());
            cudaDeviceSynchronize();
        } else // for normal and pinned
        {
            updateGridKernel<<<numBlocks, blockSize>>>(d_currentGrid, d_nextGrid, GRID_WIDTH(), GRID_HEIGHT());
            cudaDeviceSynchronize();
            cudaMemcpy(h_grid, d_currentGrid, byteSize, cudaMemcpyDeviceToHost);
        }

        // swap curr and next generation
        swap(d_currentGrid, d_nextGrid);

        // stop time
        auto endTime = clock::now();
        totalTime += chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
        numGenerations++;

        // print for every 100 generation and reset time
        if (numGenerations % 100 == 0)
        {
            cout << "100 generations took " << totalTime << " microsecs with "
                 << threadsPerBlock << " threads per block using " 
                 << MEMORY_TYPE << " memory allocation." << endl;
            totalTime = 0;
        }

        // render curr state
        window.clear(Color::Black);
        RectangleShape cell(Vector2f(PIXEL_SIZE - 1, PIXEL_SIZE - 1));
        cell.setFillColor(Color::White);

        // draw alive cells
        for (int y = 0; y < GRID_HEIGHT(); y++)
        {
            for (int x = 0; x < GRID_WIDTH(); x++)
            {
                if (h_grid[y * GRID_WIDTH() + x])
                {
                    cell.setPosition(x * PIXEL_SIZE, y * PIXEL_SIZE);
                    window.draw(cell);
                }
            }
        }

        window.display();
    }

    // cleanup
    if (MEMORY_TYPE == "MANAGED")
    {
        cudaFree(d_currentGrid);
        cudaFree(d_nextGrid);
    } else
    {
        cudaFree(d_currentGrid);
        cudaFree(d_nextGrid);
        if (MEMORY_TYPE == "PINNED")
        {
            cudaFreeHost(h_grid);
        } else
        {
            delete[] h_grid;
        }
    }

    return 0;
}