#pragma once

struct Params {
    size_t gridSize;
    size_t timeGridSize;
    double Lx;
    double Ly;
    double Lz;
    double T;
    bool logGrids = false;
    bool logErrors = false;

    double timeStep;

    Params(int argc, char *argv[]) {
        assert(argc >= 7);

        gridSize     = std::stoi(argv[1]);
        timeGridSize = std::stoi(argv[2]);
        Lx           = std::stod(argv[3]);
        Ly           = std::stod(argv[4]);
        Lz           = std::stod(argv[5]);
        T            = std::stod(argv[6]);

        for (size_t logParamsIdx = 8; logParamsIdx <= 9; ++logParamsIdx) {
            if (argc == logParamsIdx) {
                logErrors = (LOG_ERRORS == argv[logParamsIdx - 1]);
                logGrids  = (LOG_GRIDS  == argv[logParamsIdx - 1]);
            }
        }

        timeStep = T / (double) timeGridSize;
    }

    friend std::ostream &operator<<(std::ostream &_stream, Params const &params) {
        _stream << "Params(gridSize=" << params.gridSize
        << ", timeGridSize=" << params.timeGridSize
        << ", Lx=" << params.Lx
        << ", Ly=" << params.Ly
        << ", Lz=" << params.Lz
        << ", T=" << params.T << ')';
        return _stream;
    }

private:
    const std::string LOG_GRIDS = "log_grids";
    const std::string LOG_ERRORS = "log_errors";
};