#ifndef PID_PANDORA_REFERENCE_H
#define PID_PANDORA_REFERENCE_H

#include "pid_jet_resolution_summary.h"
#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pid_pandora_reference {

struct Ratio {
    Long64_t entries = 0, invalid_energy = 0, invalid_angle = 0;
    std::array<double, 102> contents{}, sumw2{};
    std::array<double, 4> stats{};
};

struct Bar {
    double efficiency = 0., purity = 0.;
};

struct Snapshot {
    std::string path;
    std::array<std::array<pid_jer::Point, 15>, 5> points{};
    std::array<std::array<Ratio, 3>, 5> ratios{};
    std::array<std::array<Bar, 5>, 5> bars{};
};

inline int energyIndex(int energy) {
    switch (energy) {
        case 40: return 0;
        case 91: return 1;
        case 200: return 2;
        case 350: return 3;
        case 500: return 4;
        default: throw std::runtime_error("Unsupported Pandora energy: " + std::to_string(energy));
    }
}

namespace detail {

class Lines {
public:
    explicit Lines(const std::string& path) : path_(path), input_(path.c_str()) {
        if (!input_) fail("Cannot open snapshot");
        input_.imbue(std::locale::classic());
    }

    bool next(std::string& value) {
        while (std::getline(input_, value)) {
            ++line_;
            const auto comment = value.find('#');
            if (comment != std::string::npos) value.erase(comment);
            if (value.find_first_not_of(" \t\r\n") != std::string::npos) return true;
        }
        if (!input_.eof()) fail("Cannot read snapshot");
        return false;
    }

    void require(std::string& value, const std::string& expected) {
        if (!next(value)) failAt(line_ + 1, "Unexpected EOF; expected " + expected);
    }

    void fail(const std::string& message) const {
        failAt(line_ ? line_ : 1, message);
    }

    void missingEnd() const { failAt(line_ + 1, "Missing END record"); }

private:
    void failAt(std::size_t line, const std::string& message) const {
        throw std::runtime_error(path_ + ":" + std::to_string(line) + ": " + message);
    }

    std::string path_;
    std::ifstream input_;
    std::size_t line_ = 0;
};

inline void noExtra(std::istringstream& row, const Lines& lines) {
    std::string extra;
    if (row >> extra) lines.fail("Unexpected token: " + extra);
}

inline void finite(double value, const Lines& lines) {
    if (!std::isfinite(value)) lines.fail("Non-finite numeric value");
}

inline int checkedEnergy(int energy, const Lines& lines) {
    try {
        return energyIndex(energy);
    } catch (const std::runtime_error& error) {
        lines.fail(error.what());
    }
    return -1; // fail() always throws.
}

inline Ratio readRatio(Lines& lines) {
    Ratio result{};
    std::array<bool, 102> found{};
    std::string text, tag;
    for (int record = 0; record < 102; ++record) {
        lines.require(text, "BIN");
        std::istringstream row(text);
        row.imbue(std::locale::classic());
        int bin = -1;
        double content = 0., sumw2 = 0.;
        if (!(row >> tag >> bin >> content >> sumw2) || tag != "BIN")
            lines.fail("Expected BIN index content sumw2");
        noExtra(row, lines);
        if (bin < 0 || bin >= 102) lines.fail("Unsupported BIN index");
        if (found[bin]) lines.fail("Duplicate BIN index");
        finite(content, lines); finite(sumw2, lines);
        if (content < 0. || sumw2 < 0.) lines.fail("Negative BIN content or Sumw2");
        result.contents[bin] = content; result.sumw2[bin] = sumw2;
        found[bin] = true;
    }
    lines.require(text, "STATS");
    std::istringstream stats(text);
    stats.imbue(std::locale::classic());
    if (!(stats >> tag >> result.stats[0] >> result.stats[1]
                >> result.stats[2] >> result.stats[3]) || tag != "STATS")
        lines.fail("Expected STATS sumw sumw2 sumwx sumwx2");
    noExtra(stats, lines);
    for (double value : result.stats) finite(value, lines);
    lines.require(text, "END_RATIO");
    std::istringstream end(text);
    end.imbue(std::locale::classic());
    if (!(end >> tag) || tag != "END_RATIO") lines.fail("Expected END_RATIO");
    noExtra(end, lines);
    return result;
}

} // namespace detail

inline Snapshot load(const std::string& path) {
    static_assert(pid_jer::kBins == 13, "Pandora snapshot requires 13 angular bins");
    detail::Lines lines(path);
    std::string text, tag;
    lines.require(text, "PANDORA_REFERENCE 1");
    std::istringstream magic(text);
    magic.imbue(std::locale::classic());
    int version = 0;
    if (!(magic >> tag >> version) || tag != "PANDORA_REFERENCE" || version != 1)
        lines.fail("Expected PANDORA_REFERENCE 1");
    detail::noExtra(magic, lines);

    Snapshot result{};
    result.path = path;
    std::array<std::array<bool, 15>, 5> points{};
    std::array<std::array<bool, 3>, 5> ratios{};
    std::array<std::array<bool, 5>, 5> bars{};
    bool ended = false;
    while (lines.next(text)) {
        std::istringstream row(text);
        row.imbue(std::locale::classic());
        row >> tag;
        if (tag == "POINT") {
            int energy = 0, index = -1, valid = -1;
            pid_jer::Point point{};
            if (!(row >> energy >> index >> point.angle_bin >> point.low >> point.high
                      >> point.entries >> point.invalid_entries >> point.underflow >> point.overflow
                      >> point.mean90 >> point.rms90 >> point.resolution_percent
                      >> point.error_percent >> valid))
                lines.fail("Malformed POINT record");
            detail::noExtra(row, lines);
            const int e = detail::checkedEnergy(energy, lines);
            if (index < 0 || index >= 15) lines.fail("Unsupported POINT index");
            if (points[e][index]) lines.fail("Duplicate POINT energy/index");
            if (valid != 0 && valid != 1) lines.fail("POINT valid must be 0 or 1");
            point.valid = valid != 0;
            for (double value : {point.low, point.high, point.underflow, point.overflow,
                                 point.mean90, point.rms90, point.resolution_percent,
                                 point.error_percent}) detail::finite(value, lines);
            if (point.entries < 0 || point.invalid_entries < 0 || point.underflow < 0. ||
                point.overflow < 0. || point.rms90 < 0. || point.resolution_percent < 0. ||
                point.error_percent < 0. || (point.valid && (point.entries < 5 ||
                point.invalid_entries != 0 || point.underflow != 0. || point.overflow != 0. ||
                !(point.mean90 > 0.))))
                lines.fail("Invalid POINT statistics");
            const int angle_bin = index < 13 ? index : (index == 13 ? -1 : -2);
            const double low = index < 13 ? pid_jer::kEdges[index] : (index == 13 ? 0. : .7);
            const double high = index < 13 ? pid_jer::kEdges[index + 1] : (index == 13 ? .7 : .98);
            if (point.angle_bin != angle_bin || std::fabs(point.low - low) > 1e-12 ||
                std::fabs(point.high - high) > 1e-12)
                lines.fail("POINT angle_bin or boundaries are incompatible");
            result.points[e][index] = point;
            points[e][index] = true;
        } else if (tag == "RATIO") {
            int energy = 0, region = -1;
            Long64_t entries = 0, invalid_energy = 0, invalid_angle = 0;
            if (!(row >> energy >> region >> entries >> invalid_energy >> invalid_angle))
                lines.fail("Malformed RATIO record");
            detail::noExtra(row, lines);
            const int e = detail::checkedEnergy(energy, lines);
            if (region < 0 || region >= 3) lines.fail("Unsupported RATIO region");
            if (ratios[e][region]) lines.fail("Duplicate RATIO energy/region");
            if (entries < 0 || invalid_energy < 0 || invalid_angle < 0)
                lines.fail("Negative RATIO event count");
            Ratio ratio = detail::readRatio(lines);
            ratio.entries = entries; ratio.invalid_energy = invalid_energy;
            ratio.invalid_angle = invalid_angle;
            result.ratios[e][region] = ratio;
            ratios[e][region] = true;
        } else if (tag == "BAR") {
            int energy = 0, particle = -1;
            Bar bar{};
            if (!(row >> energy >> particle >> bar.efficiency >> bar.purity))
                lines.fail("Malformed BAR record");
            detail::noExtra(row, lines);
            const int e = detail::checkedEnergy(energy, lines);
            if (particle < 0 || particle >= 5) lines.fail("Unsupported BAR particle");
            if (bars[e][particle]) lines.fail("Duplicate BAR energy/particle");
            detail::finite(bar.efficiency, lines); detail::finite(bar.purity, lines);
            if (bar.efficiency < 0. || bar.efficiency > 1. || bar.purity < 0. || bar.purity > 1.)
                lines.fail("BAR efficiency/purity must be in [0,1]");
            result.bars[e][particle] = bar;
            bars[e][particle] = true;
        } else if (tag == "END") {
            detail::noExtra(row, lines);
            ended = true;
            break;
        } else {
            lines.fail("Unknown record: " + tag);
        }
    }
    if (!ended) lines.missingEnd();
    for (int e = 0; e < 5; ++e) {
        for (int i = 0; i < 15; ++i)
            if (!points[e][i]) lines.fail("Missing POINT record");
        for (int r = 0; r < 3; ++r)
            if (!ratios[e][r]) lines.fail("Missing RATIO record");
        for (int p = 0; p < 5; ++p)
            if (!bars[e][p]) lines.fail("Missing BAR record");
    }
    if (lines.next(text)) lines.fail("Unexpected record after END");
    return result;
}

} // namespace pid_pandora_reference

#endif
