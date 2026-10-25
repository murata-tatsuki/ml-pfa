#ifndef JER_LOSS_CORRELATIONS_H
#define JER_LOSS_CORRELATIONS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace jer_epoch_plot {
namespace loss_correlation {
namespace detail {

// One-based average ranks make exact ties contribute equally. Working with
// ranks also avoids overflow when the original finite values are very large.
inline std::vector<long double> averageRanks(const std::vector<double>& values) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return values[left] < values[right];
    });
    std::vector<long double> ranks(values.size());
    for (std::size_t begin = 0; begin < order.size();) {
        std::size_t end = begin + 1;
        while (end < order.size() && values[order[end]] == values[order[begin]]) ++end;
        const long double rank = (static_cast<long double>(begin) +
                                  static_cast<long double>(end) + 1.L) / 2.L;
        for (std::size_t index = begin; index < end; ++index) ranks[order[index]] = rank;
        begin = end;
    }
    return ranks;
}

} // namespace detail

// Unweighted Spearman correlation, i.e. Pearson correlation of average ranks.
// A pair is omitted when either value is NaN or infinite. Fewer than three
// finite pairs, or a constant finite input, give NaN rather than a fake zero.
// Input lengths must agree; uncertainty/error bars are not correlation weights.
inline double spearman(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size())
        throw std::invalid_argument("Spearman inputs must have the same number of values");
    std::vector<double> finiteX, finiteY;
    finiteX.reserve(x.size()); finiteY.reserve(y.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        if (std::isfinite(x[index]) && std::isfinite(y[index])) {
            finiteX.push_back(x[index]); finiteY.push_back(y[index]);
        }
    }
    const double unavailable = std::numeric_limits<double>::quiet_NaN();
    if (finiteX.size() < 3) return unavailable;
    const auto ranksX = detail::averageRanks(finiteX);
    const auto ranksY = detail::averageRanks(finiteY);
    const long double mean = (static_cast<long double>(finiteX.size()) + 1.L) / 2.L;
    long double covariance = 0, varianceX = 0, varianceY = 0;
    for (std::size_t index = 0; index < ranksX.size(); ++index) {
        const long double dx = ranksX[index] - mean, dy = ranksY[index] - mean;
        covariance += dx * dy; varianceX += dx * dx; varianceY += dy * dy;
    }
    if (varianceX == 0 || varianceY == 0) return unavailable;
    const long double rho = covariance / std::sqrt(varianceX * varianceY);
    // Guard the exact mathematical bounds against final floating-point rounding.
    return static_cast<double>(std::max(-1.L, std::min(1.L, rho)));
}

} // namespace loss_correlation
} // namespace jer_epoch_plot

#endif
