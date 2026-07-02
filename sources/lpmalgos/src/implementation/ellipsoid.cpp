#pragma once

#include "ellipsoid.h"
#include <tuple>

namespace lpmalgos
{

static const double pi = std::acos(-1);

Matrix orthogonolize_axes(Matrix nei, bool sort)
{
    Location e1{1, 0, 0};
    Location e2{0, 1, 0};
    Location e3{0, 0, 1};

    // orthogonalize rows (assure numerical stability)
    Location u1 = nei.row(0);
    Location v2 = nei.row(1);
    Location v3 = nei.row(2);

    Location u2 = v2 - v2.dot(u1) * u1 / u1.dot(u1);
    Location u3 = v3 - v3.dot(u2) * u2 / u2.dot(u2) - v3.dot(u1) * u1 / u1.dot(u1);

    if (!sort) {
        Matrix r = Matrix::Zero(3, 3);
        r.row(0) = u1;
        r.row(1) = u2;
        r.row(2) = u3;
        return r;
    }

    double d1 = (u1 / u1.norm() - e1).norm();
    double d2 = (u1 / u1.norm() + e1).norm();

    if (std::fabs(d1 - d2) > 1e-8) {
        nei.row(0) = d1 > d2 ? (-u1) : u1;
    } else {
        d1 = (u1 / u1.norm() - e2).norm();
        d2 = (u1 / u1.norm() + e2).norm();
        if (std::fabs(d1 - d2) > 1e-8) {
            nei.row(0) = d1 > d2 ? (-u1) : u1;
        } else {
            d1 = (u1 / u1.norm() - e3).norm();
            d2 = (u1 / u1.norm() + e3).norm();
            nei.row(0) = d1 > d2 ? (-u1) : u1;
        }
    }

    d1 = (u2 / u2.norm() - e2).norm();
    d2 = (u2 / u2.norm() + e2).norm();

    if (std::fabs(d1 - d2) > 1e-8) {
        nei.row(1) = d1 > d2 ? (-u2) : u2;
    } else {
        d1 = (u2 / u2.norm() - e1).norm();
        d2 = (u2 / u2.norm() + e1).norm();
        if (std::fabs(d1 - d2) > 1e-8) {
            nei.row(1) = d1 > d2 ? (-u2) : u2;
        } else {
            d1 = (u2 / u2.norm() - e3).norm();
            d2 = (u2 / u2.norm() + e3).norm();
            nei.row(1) = d1 > d2 ? (-u2) : u2;
        }
    }

    d1 = (u3 / u3.norm() - e3).norm();
    d2 = (u3 / u3.norm() + e3).norm();

    if (std::fabs(d1 - d2) > 1e-8) {
        nei.row(2) = d1 > d2 ? (-u3) : u3;
    } else {
        d1 = (u2 / u2.norm() - e1).norm();
        d2 = (u2 / u2.norm() + e1).norm();
        if (std::fabs(d1 - d2) > 1e-8) {
            nei.row(2) = d1 > d2 ? (-u3) : u3;
        } else {
            d1 = (u2 / u2.norm() - e2).norm();
            d2 = (u2 / u2.norm() + e2).norm();
            nei.row(2) = d1 > d2 ? (-u3) : u3;
        }
    }

    return nei;
}

Matrix sort_matrix3d(Matrix m, bool orthogonalize)
{
    Location u1 = m.row(0);
    Location u2 = m.row(1);
    Location u3 = m.row(2);

    Locations lines{u1, u2, u3};
    std::vector<std::tuple<double, uint8_t>> line_norms{
        std::make_tuple(u1.norm(), static_cast<unsigned char>(0)),
        std::make_tuple(u2.norm(), static_cast<unsigned char>(1)),
        std::make_tuple(u3.norm(), static_cast<unsigned char>(2))};
    std::sort(line_norms.begin(), line_norms.end());

    m.row(0) = lines[std::get<1>(line_norms[0])];
    m.row(1) = lines[std::get<1>(line_norms[1])];
    m.row(2) = lines[std::get<1>(line_norms[2])];
    if (orthogonalize) {
        m = orthogonolize_axes(m);
    }
    return m;
}

EllipsoidInfo extract_ellipsoid_info(const Matrix &other)
{
    Matrix T = sort_matrix3d(other, true);
    EllipsoidInfo info;

    Location v1 = T.row(0);
    Location v2 = T.row(1);
    Location v3 = T.row(2);

    std::vector<double> s{v1.norm(), v2.norm(), v3.norm()};
    Matrix S = Matrix::Zero(3, 3);
    S(0, 0) = info.r1 = 1 / (s[0] > 0 ? s[0] : 1);
    S(1, 1) = info.r2 = 1 / (s[1] > 0 ? s[1] : 1);
    S(2, 2) = info.r3 = 1 / (s[2] > 0 ? s[2] : 1);

    Matrix R = S * T;
    double alpha(0), beta(0), theta(0);

    if (std::fabs(R(0, 2)) < (1 - 1e-8)) {
        beta = std::asin(-R(0, 2));
    } else {
        beta = R(0, 2) < 0 ? (0.5 * pi) : (-0.5 * pi);
    }

    double csbeta = std::cos(beta);

    if (std::fabs(csbeta) > 1e-6) {
        alpha = std::atan2(R(0, 1), R(0, 0));
        theta = std::atan2(R(1, 2), R(2, 2));
    } else {
        theta = 0;
        if (beta > 0) {
            alpha = std::atan2(R(2, 1), R(2, 0));
        } else {
            alpha = std::fmod(3 * pi - std::atan2(-R(2, 1), R(2, 0)), 2 * pi);
        }
    }

    info.azimuth = alpha / pi * 180;
    info.dip = beta / pi * 180;
    info.rake = theta / pi * 180;

    info.azimuth = std::fmod(450 - info.azimuth, 360);
    info.dip = -info.dip;
    info.rake = info.rake;
    return info;
}

EllipsoidInfo::EllipsoidInfo(double r1, double r2, double r3,
        double azimuth, double dip, double rake)
    : r1(r1), r2(r2), r3(r3), azimuth(azimuth), dip(dip), rake(rake)
{
}

Ellipsoid::Ellipsoid(double r1, double r2, double r3,
        double azimuth, double dip, double rake)
{
    double alpha = (90 - azimuth) / 180 * pi;
    double beta = (-dip) / 180 * pi;
    double theta = rake / 180 * pi;

    if (r1 <= 1e-30) {
        r1 = 1;
    }

    if (r2 <= 1e-30) {
        r2 = 1;
    }

    if (r3 <= 1e-30) {
        r3 = 1;
    }

    Matrix S, rz, ry, rx;

    S << 1.0 / r1, 0, 0, 0, 1.0 / r2, 0, 0, 0, 1.0 / r3;
    rz << cos(alpha), sin(alpha), 0, -sin(alpha), cos(alpha), 0, 0, 0, 1;
    ry << cos(beta), 0, -sin(beta), 0, 1, 0, sin(beta), 0, cos(beta);
    rx << 1, 0, 0, 0, cos(theta), sin(theta), 0, -sin(theta), cos(theta);

    trans = S * ((rx * ry) * rz);
    inv_trans = trans.inverse();
}

Ellipsoid::Ellipsoid(const EllipsoidInfo &info)
    : Ellipsoid(info.r1, info.r2, info.r3, info.azimuth, info.dip, info.rake)
{
}

Ellipsoid::Ellipsoid(const Matrix &T)
{
    trans = T;
    inv_trans = T.inverse();
}

Ellipsoid::Ellipsoid()
{
    trans << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    inv_trans << 1, 0, 0, 0, 1, 0, 0, 0, 1;
}

EllipsoidInfo Ellipsoid::info() const
{
    return extract_ellipsoid_info(matrix());
}

std::string Ellipsoid::to_string() const
{
    const EllipsoidInfo info = (*this).info();
    return "{r1: " + std::to_string(info.r1) +
           ", r2: " + std::to_string(info.r2) +
           ", r3: " + std::to_string(info.r3) +
           ", azimuth: " + std::to_string(info.azimuth) +
           ", dip: " + std::to_string(info.dip) +
           ", rake: " + std::to_string(info.rake) + "} ";
}

} // namespace lpmalgos