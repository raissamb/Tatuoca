#pragma once

// Adicione exatamente este bloco aqui se não estiver:
#ifndef _MSC_VER
#define Ui8
#define Ui64 ULL
#endif

#include "machine.h"
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace lpmalgos
{

using Matrix = Eigen::Matrix3d;
using Location = Eigen::Vector3d;
using Locations = std::vector<Location>;

Matrix orthogonolize_axes(Matrix nei, bool sort = true);
Matrix sort_matrix3d(Matrix m, bool orthogonalize = true);

struct EllipsoidInfo {
    double r1, r2, r3, azimuth, dip, rake;

    EllipsoidInfo(double r1 = 0, double r2 = 0, double r3 = 0,
        double azimuth = 0, double dip = 0, double rake = 0);
};
EllipsoidInfo extract_ellipsoid_info(const Matrix &other);

class Ellipsoid
{
public:
    Ellipsoid(double r1, double r2, double r3,
        double azimuth, double dip, double rake);
    Ellipsoid(const EllipsoidInfo &info);
    Ellipsoid(const Matrix &T);
    Ellipsoid();

    inline Location forward(const Location &loc) const noexcept
    {
        return trans * loc;
    }
    inline Location backward(const Location &loc) const noexcept
    {
        return inv_trans * loc;
    }

    inline Locations forward(const Locations &locs) const noexcept
    {
        Locations result(locs.size());
        unsigned n_threads = num_threads();
#pragma omp parallel for num_threads(n_threads)
        for (int64_t i = 0; i < (int64_t) locs.size(); ++i) {
            result[i] = forward(locs[i]);
        }
        return result;
    }

    inline Locations backward(const Locations &locs) const noexcept
    {
        Locations result(locs.size());
        unsigned n_threads = num_threads();
#pragma omp parallel for num_threads(n_threads)
        for (int64_t i = 0; i < (int64_t) locs.size(); ++i) {
            result[i] = backward(locs[i]);
        }
        return result;
    }

    inline Matrix matrix() const noexcept { return trans; }
    inline Matrix inv_matrix() const noexcept { return inv_trans; }

    inline Location major_axis() const noexcept
    {
        auto T = sort_matrix3d(trans);
        return T.row(0) / std::pow(T.row(0).norm(), 2);
    }

    inline Location mid_axis() const noexcept
    {
        auto T = sort_matrix3d(trans);
        return T.row(1) / std::pow(T.row(1).norm(), 2);
    }

    inline Location minor_axis() const noexcept
    {
        auto T = sort_matrix3d(trans);
        return T.row(2) / std::pow(T.row(2).norm(), 2);
    }

    EllipsoidInfo info() const;
    std::string to_string() const;

private:
    Matrix trans, inv_trans;
};

} // namespace lpmalgos