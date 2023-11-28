#pragma once

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