#include <iostream>
#include <cassert>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <chrono>

#include "params.h"
#include "grid.h"
#include "grid_applier.h"

static const std::string LOG_DIR = "./run_logs";

static const double aSquared = 1 / (4 * pi * pi);

struct Error {
    void updateErrors(double curError) {
        updateMaxError(curError);
        updateAvgError(curError);
    }

    void updateMaxError(double curError) {
        maxError = std::max(maxError, curError);
    }

    void updateAvgError(double curError) {
        ++nIters;
        summedError += curError;
    }

    double getMaxError() const {
        return maxError;
    }

    double getAvgError() const {
        return nIters > 0 ? summedError / (double) nIters : 0.;
    }

    void clear() {
        nIters = 0;
        summedError = 0.;
        maxError = 0.;
    };

private:
    size_t nIters{};
    double summedError{};
    double maxError{};
};

Error initGridT1(const Grid& gridT0, Grid& gridT1, const GridApplier& applier, double timeStep) {
    Error error = Error();
    for (size_t i = 1; i < gridT1.xSize - 1; ++i) {
        for (size_t j = 1; j < gridT1.ySize - 1; ++j) {
            for (size_t k = 1; k < gridT1.zSize - 1; ++k) {
                double uT0 = gridT0.get(i, j, k);
                double uT1 = uT0 + aSquared * timeStep * timeStep / 2. * gridT0.getLaplace(i, j, k);
                gridT1.set(i, j, k, uT1);
                double curError = std::fabs(uT1 - applier.applyOnGridPoint(gridT1, i, j, k, timeStep));
                error.updateErrors(curError);
            }
        }
    }
    return error;
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
    Params params = Params(argc, argv);
    std::cout << "Running task with params:\n" << params << std::endl;

    bool logErrors = params.logErrors;
    bool logGrids = params.logGrids;
    if (logErrors || logGrids) {
        struct stat st = {0};
        if (stat(LOG_DIR.c_str(), &st) == -1) {
            mkdir(LOG_DIR.c_str(), 0700);
        }
    }

    double timeStep = params.timeStep;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    GridApplier gridApplier(params);
    Grid prevGrid(params);
    Grid curGrid(params);

    gridApplier.applyOnGrid(prevGrid, 0);
    gridApplier.applyOnGrid(curGrid, timeStep);

    logResults(prevGrid, {}, 0, logErrors, logGrids);
    Error error = initGridT1(prevGrid, curGrid, gridApplier, params.timeStep);
    logResults(curGrid, error, 1, logErrors, logGrids);

    Grid newGrid(params);
    for (size_t t = 2; t < params.timeGridSize; ++t) {
        newGrid.clearGrid();
        error.clear();
        #pragma omp parallel for
        for (size_t i = 0; i < newGrid.xSize; ++i) {
            for (size_t j = 1; j < newGrid.ySize - 1; ++j) {
                for (size_t k = 1; k < newGrid.zSize - 1; ++k) {
                    double uValue = timeStep * timeStep * aSquared * curGrid.getLaplace(i, j, k)
                                  + 2 * curGrid.get(i, j, k)
                                  - prevGrid.get(i, j, k);
                    newGrid.set(i, j, k, uValue);

                    double curError = std::fabs(uValue - gridApplier.applyOnGridPoint(newGrid, i, j, k, timeStep * t));
                    error.updateErrors(curError);
                }
            }
        }
        logResults(newGrid, error, t, logErrors, logGrids);

        prevGrid.moveValues(curGrid);
        curGrid.moveValues(newGrid);
    }
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    long taskWorkTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Task took " << taskWorkTime << " ms (~"
              << taskWorkTime / params.timeGridSize << "ms per iteration)" << std::endl;

    std::cout << "Avg error = " <<  error.getAvgError() << std::endl;
    std::cout << "Max error = " <<  error.getMaxError() << std::endl;

    std::cout << "Task finished successfully!" << std::endl;
    return 0;
}