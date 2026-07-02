#pragma once

#include "neighborhood.h"

namespace lpmalgos
{

Neighborhood::Neighborhood(const Locations &locs,
                           const Ellipsoid &ani)
    : Neighborhood(locs.size(), [&locs, &ani](size_t i)
                                { return ani.forward(locs[i]); })
{
    use_anisotropy = true;
    anisotropy = ani;
}

Neighborhood::Neighborhood(const Locations &locs)
    : Neighborhood(locs.size(), [&locs](size_t i) { return locs[i]; })
{
    use_anisotropy = false;
}

Neighborhood::Neighborhood(size_t size, const location_function &func)
{
    for (size_t i = 0; i < size; ++i) {
        auto loc = func(i);
        Point_cloud::Point point{loc.x(), loc.y(), loc.z()};
        cloud.pts.push_back(point);
    }

    nanoflann::KDTreeSingleIndexAdaptorParams params;
    kdtree = std::make_unique<kd_tree>(3, cloud, params);
}

#pragma warning(push)
#pragma warning(disable: 4267)

std::vector<size_t> Neighborhood::find_neighbors(const Location &loc,
                                                 size_t max_neighbors) const
{
    if (max_neighbors > size()) max_neighbors = size();

    double query_pt[3];
    if (use_anisotropy) {
        Location loc2 = anisotropy.forward(loc);
        query_pt[0] = loc2.x();
        query_pt[1] = loc2.y();
        query_pt[2] = loc2.z();
    } else {
        query_pt[0] = loc.x();
        query_pt[1] = loc.y();
        query_pt[2] = loc.z();
    }

    std::vector<size_t> neis(max_neighbors);
    auto dists = std::make_unique<double[]>(max_neighbors);
    nanoflann::KNNResultSet<double> resultSet(max_neighbors);
    resultSet.init(neis.data(), dists.get());
    (*kdtree).findNeighbors(resultSet, &query_pt[0],
                            nanoflann::SearchParameters());

    return neis;
}

#pragma warning(pop)

std::vector<size_t> Neighborhood::find_neighbors(const Location &loc,
                                                 double max_radius) const
{
    max_radius *= max_radius;

    double query_pt[3];
    if (use_anisotropy) {
        Location loc2 = anisotropy.forward(loc);
        query_pt[0] = loc2.x();
        query_pt[1] = loc2.y();
        query_pt[2] = loc2.z();
    } else {
        query_pt[0] = loc.x();
        query_pt[1] = loc.y();
        query_pt[2] = loc.z();
    }

    std::vector<nanoflann::ResultItem<size_t, double>> ind;
    (*kdtree).radiusSearch(&query_pt[0], max_radius, ind,
                           nanoflann::SearchParameters());

    std::vector<size_t> neis(ind.size());
    for (size_t i = 0; i < ind.size(); i++) {
        neis[i] = ind[i].first;
    }

    return neis;
}

size_t Neighborhood::nearest_neighbor(const Location &loc)
{
    // Usando static_cast para garantir que o compilador chame a versão (loc, size_t)
    auto neis = find_neighbors(loc, static_cast<size_t>(1));
    return neis[0];
}

} // namespace lpmalgos