#pragma once

#include <cmath>

static const double pi = atan(1) * 4;

class GridApplier {
    const double Lx;
    const double Ly;
    const double Lz;

    const double aT;

    double u(double x, double y, double z, double t) const {
        return sin(2 * pi * x / Lx) * sin(pi * y / Ly) * sin(pi * z / Lz) * cos(aT * t + 2 * pi);
    }

public:
    explicit GridApplier(Params params) :
            Lx(params.Lx),
            Ly(params.Ly),
            Lz(params.Lz),
            aT(0.5 * sqrt(4. / (Lx * Lx) + 1. / (Ly * Ly) + 1. / (Lz * Lz))) {}

    double applyOnGridPoint(const Grid &grid, size_t i, size_t j, size_t k, double t) const {
        return u((double) i * grid.xStep,
                 (double) j * grid.yStep,
                 (double) k * grid.zStep,
                 t);
    }

    void applyOnGrid(Grid &grid, double t) const {
        #pragma omp parallel for
        for (size_t i = 0; i < grid.xSize; ++i) {
            for (size_t j = 0; j < grid.ySize; ++j) {
                for (size_t k = 0; k < grid.zSize; ++k) {
                    grid.set(i, j, k, applyOnGridPoint(grid, i, j, k, t));
                }
            }
        }
    }
};