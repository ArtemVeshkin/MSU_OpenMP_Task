#include <iostream>
#include <cassert>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <mpi.h>

#include "params.h"
#include "grid.h"
#include "grid_applier.h"
#include "error.h"


#define MAIN_THREAD 0

static const std::string LOG_DIR = "./run_logs";

static const double aSquared = 1 / (4 * pi * pi);

Error initGridT1(const Grid& gridT0, Grid& gridT1, const GridApplier& applier, double timeStep, size_t *coordShift) {
    Error error = Error();
    #pragma omp parallel for
    for (size_t i = 1; i < gridT1.xSize - 1; ++i) {
        for (size_t j = 1; j < gridT1.ySize - 1; ++j) {
            for (size_t k = 1; k < gridT1.zSize - 1; ++k) {
                double uT0 = gridT0.get(i, j, k);
                double uT1 = uT0 + aSquared * timeStep * timeStep / 2. * gridT0.getLaplace(i, j, k);
                gridT1.set(i, j, k, uT1);

                double analyticalValue = applier.applyOnGridPoint(
                        gridT1,
                        i + coordShift[0] - 1,
                        j + coordShift[1] - 1,
                        k + coordShift[2] - 1,
                        timeStep);
                double curError = std::fabs(uT1 - analyticalValue);
                error.updateErrors(curError);
            }
        }
    }
    return error;
}

MPI_Datatype createSubarrayType(size_t *paddedBlockSize, size_t *subSizes) {
    constexpr int start[3] = {0, 0, 0};
    MPI_Datatype subarrayType;
    MPI_Type_create_subarray(3, reinterpret_cast<const int *>(paddedBlockSize), reinterpret_cast<const int *>(subSizes), start, MPI_ORDER_C, MPI_DOUBLE, &subarrayType);
    MPI_Type_commit(&subarrayType);
    return subarrayType;
}

void logResults(const Grid& grid, Error error, size_t timeStep, bool logErrors, bool logGrids) {
    if (logErrors) {
        std::ofstream maxErrorFile(LOG_DIR + "/max_errors.txt", std::ios::app);
        std::ofstream avgErrorFile(LOG_DIR + "/avg_errors.txt", std::ios::app);
        maxErrorFile << error.getMaxError() << std::endl;
        avgErrorFile << error.getAvgError() << std::endl;
        maxErrorFile.close();
        avgErrorFile.close();
    }

    if (logGrids) {
        mkdir((LOG_DIR + "/grids").c_str(), 0700);
        std::stringstream gridLogPath;
        gridLogPath << LOG_DIR << "/grids/" << timeStep << "_grid.txt";
        std::ofstream gridFile(gridLogPath.str());

        gridFile << grid.xSize << std::endl;
        gridFile << grid.ySize << std::endl;
        gridFile << grid.zSize << std::endl;

        for (size_t i = 0; i < grid.xSize; ++i) {
            for (size_t j = 0; j < grid.ySize; ++j) {
                for (size_t k = 0; k < grid.zSize; ++k) {
                    gridFile << grid.get(i, j, k) << std::endl;
                }
            }
        }
        gridFile.close();
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int thread, threadSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &thread);
    MPI_Comm_size(MPI_COMM_WORLD, &threadSize);

    Params params = Params(argc, argv);
    if (thread == MAIN_THREAD) {
        std::cout << "Running task with params:\n" << params << std::endl;
    }

    bool logErrors = params.logErrors;
    bool logGrids  = params.logGrids;
    if ((thread == MAIN_THREAD) && (logErrors || logGrids)) {
        struct stat st = {0};
        if (stat(LOG_DIR.c_str(), &st) == -1) {
            mkdir(LOG_DIR.c_str(), 0700);
        }
    }

    int dimThreads[3] = {0, 0, 0};
    MPI_Dims_create(threadSize, 3, dimThreads);

    MPI_Comm gridComm;
    int periods[3] = {true, true, true};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dimThreads, periods, false, &gridComm);
    int coords[3];
    MPI_Cart_coords(gridComm, thread, 3, coords);

    size_t leftBorders[3];
    size_t rightBorders[3];
    size_t blockSize[3];
    size_t paddedBlockSize[3];

    for (size_t dim = 0; dim < 3; ++dim) {
        size_t cleanBlockSize = params.gridSize / dimThreads[dim];

        leftBorders[dim]  = cleanBlockSize * coords[dim];
        rightBorders[dim] = cleanBlockSize * (coords[dim] + 1);
        if (coords[dim] == dimThreads[dim] - 1) rightBorders[dim] += params.gridSize % dimThreads[dim];

        blockSize[dim] = rightBorders[dim] - leftBorders[dim];
        paddedBlockSize[dim] = blockSize[dim] + 2;
    }

    double startTime = MPI_Wtime();

    GridApplier gridApplier(params);
    Grid prevGrid(params, paddedBlockSize[0], paddedBlockSize[1], paddedBlockSize[2]);
    Grid curGrid(params, paddedBlockSize[0], paddedBlockSize[1], paddedBlockSize[2]);

    double timeStep = params.timeStep;
    gridApplier.applyOnGrid(prevGrid, 0, leftBorders);
    gridApplier.applyOnGrid(curGrid, timeStep, leftBorders);

    if (thread == MAIN_THREAD) {
        logResults(prevGrid, {}, 0, logErrors, logGrids);
    }

    Error error = initGridT1(prevGrid, curGrid, gridApplier, params.timeStep, leftBorders);

    if (thread == MAIN_THREAD) {
        logResults(curGrid, error, 1, logErrors, logGrids);
    }

    size_t xSubSizes[3] = {1, blockSize[1], blockSize[2]};
    size_t ySubSizes[3] = {blockSize[0], 1, blockSize[2]};
    size_t zSubSizes[3] = {blockSize[0], blockSize[1], 1};
    MPI_Datatype subarrayTypes[3] = {
            createSubarrayType(paddedBlockSize, xSubSizes),
            createSubarrayType(paddedBlockSize, ySubSizes),
            createSubarrayType(paddedBlockSize, zSubSizes)
    };

    Grid newGrid(params, paddedBlockSize[0], paddedBlockSize[1], paddedBlockSize[2]);
    for (size_t t = 2; t < params.timeGridSize; ++t) {

        for (int dim = 0; dim < 3; ++dim) {
            int prevThread, curThread, nextThread;
            MPI_Cart_shift(gridComm, dim, 1, &curThread, &nextThread);
            MPI_Cart_shift(gridComm, dim, -1, &curThread, &prevThread);

            MPI_Request request;
            MPI_Isend(prevGrid.getPtr(1, 1, 1), 1, subarrayTypes[dim], prevThread, 2 * dim, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);

            double* outRightBorderPtr = prevGrid.getPtr(
                    dim == 0 ? blockSize[dim] : 1,
                    dim == 1 ? blockSize[dim] : 1,
                    dim == 2 ? blockSize[dim] : 1
            );
            MPI_Isend(outRightBorderPtr, 1, subarrayTypes[dim], nextThread, 2 * dim + 1, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);

            double* inLeftBorderPtr = prevGrid.getPtr(
                    dim == 0 ? 0 : 1,
                    dim == 1 ? 0 : 1,
                    dim == 2 ? 0 : 1
            );
            double* inRightBorderPtr = prevGrid.getPtr(
                    dim == 0 ? paddedBlockSize[dim] : 1,
                    dim == 1 ? paddedBlockSize[dim] : 1,
                    dim == 2 ? paddedBlockSize[dim] : 1
            );
            MPI_Recv(inLeftBorderPtr, 1, subarrayTypes[dim], prevThread, 2 * dim + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(inRightBorderPtr, 1, subarrayTypes[dim], nextThread, 2 * dim, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        newGrid.clearGrid();
        error.clear();
        #pragma omp parallel for
        for (size_t i = 1; i < newGrid.xSize; ++i) {
            for (size_t j = 1; j < newGrid.ySize - 1; ++j) {
                for (size_t k = 1; k < newGrid.zSize - 1; ++k) {
                    double uValue = timeStep * timeStep * aSquared * curGrid.getLaplace(i, j, k)
                                  + 2 * curGrid.get(i, j, k)
                                  - prevGrid.get(i, j, k);
                    newGrid.set(i, j, k, uValue);


                     double analyticalValue = gridApplier.applyOnGridPoint(
                        newGrid,
                        i + leftBorders[0] - 1,
                        j + leftBorders[1] - 1,
                        k + leftBorders[2] - 1,
                        timeStep * t);
                    double curError = std::fabs(uValue - analyticalValue);
                    error.updateErrors(curError);
                }
            }
        }
        logResults(newGrid, error, t, logErrors, logGrids);

        prevGrid.moveValues(curGrid);
        curGrid.moveValues(newGrid);
    }
    double endTime = MPI_Wtime();
    double taskWorkTime = endTime - startTime;
    double maxTaskWorkTime;
    MPI_Reduce(&taskWorkTime, &maxTaskWorkTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double curMaxError = error.getMaxError();
    double totalMaxError;
    MPI_Reduce(&curMaxError, &totalMaxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (thread == MAIN_THREAD) {
        std::cout << "Task took " << 1000 * maxTaskWorkTime << " ms (~"
              << 1000 * maxTaskWorkTime / (double) params.timeGridSize << " ms per iteration)" << std::endl;

        std::cout << "Max error = " <<  totalMaxError << std::endl;
        std::cout << "Task finished successfully!" << std::endl;
    }

    for (MPI_Datatype& subarrayType : subarrayTypes) {
        MPI_Type_free(&subarrayType);
    }
    MPI_Finalize();
    return 0;
}