#pragma once

class Grid {
    double *gridValues;

    size_t getIdx(size_t x, size_t y, size_t z) const {
        return x * (ySize * zSize) + y * zSize + z;
    }

public:
    size_t xSize, ySize, zSize;
    double xStep, yStep, zStep;

    Grid(const Params& params, size_t xSize, size_t ySize, size_t zSize) {
        this->xSize = xSize;
        this->ySize = ySize;
        this->zSize = zSize;
        xStep = params.Lx / (double) xSize;
        yStep = params.Ly / (double) ySize;
        zStep = params.Lz / (double) zSize;
        gridValues = new double[xSize * ySize * zSize];
    }

    explicit Grid(const Params& params) : Grid(params, params.gridSize, params.gridSize, params.gridSize) {}

    void clearGrid() {
        #pragma omp parallel for
        for (size_t i = 0; i < xSize * ySize * zSize; ++i) {
            gridValues[i] = 0.;
        }
    }

    double get(size_t i, size_t j, size_t k) const {
        return gridValues[getIdx(i, j, k)];
    }

    double* getPtr(size_t i, size_t j, size_t k) {
        return &gridValues[getIdx(i, j, k)];
    }

    void set(size_t i, size_t j, size_t k, double value) {
        gridValues[getIdx(i, j, k)] = value;
    }

    double getLaplace(size_t i, size_t j, size_t k) const {
        bool isBorderI = (i == 0 || i == xSize - 1);
        double dX = (get((isBorderI ? xSize - 1 : i) - 1, j, k)
                     - 2 * get(i, j, k)
                     + get((isBorderI ? 0 : i) + 1, j, k)) / (xStep * xStep);
        double dY = (get(i, j - 1, k) - 2 * get(i, j, k) + get(i, j + 1, k)) / (yStep * yStep);
        double dZ = (get(i, j, k - 1) - 2 * get(i, j, k) + get(i, j, k + 1)) / (zStep * zStep);

        return dX + dY + dZ;
    }

    void moveValues(const Grid& newGrid) {
        assert(xSize == newGrid.xSize);
        assert(ySize == newGrid.ySize);
        assert(zSize == newGrid.zSize);
        #pragma omp parallel for
        for (size_t i = 0; i < xSize * ySize * zSize; ++i) {
            gridValues[i] = newGrid.gridValues[i];
        }
    }

    ~Grid() {
        delete[] gridValues;
    }
};