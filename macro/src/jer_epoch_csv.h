#ifndef JER_EPOCH_CSV_H
#define JER_EPOCH_CSV_H

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace jer_epoch_plot {

struct CsvRow {
    std::string model, run, method, readout, region, digest;
    int epoch = 0, energy = 0, angleBin = 0;
    long long events = 0, entries = 0, expectedFiles = 0, collectedFiles = 0;
    double low = 0, high = 0, resolution = 0, error = 0, betaThreshold = 0;
};

namespace csv_detail {

inline std::string trim(const std::string &value) {
    std::size_t begin = 0, end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(begin, end - begin);
}

// RFC 4180-style fields, including escaped quotes, embedded newlines and CRLF.
// A record is returned even when its final line has no newline.
inline bool readRecord(std::istream &input, std::vector<std::string> &fields,
                       std::size_t &line, const std::string &path) {
    fields.clear();
    std::string field;
    bool quoted = false, afterQuote = false, started = false;
    for (;;) {
        const int next = input.get();
        if (next == std::char_traits<char>::eof()) {
            if (input.bad()) throw std::runtime_error(path + ": failed reading CSV");
            if (quoted) throw std::runtime_error(path + ": unterminated quoted field near line " + std::to_string(line));
            if (!started && fields.empty() && field.empty()) return false;
            fields.push_back(field);
            return true;
        }
        const char ch = static_cast<char>(next);
        started = true;
        if (quoted) {
            if (ch == '"') {
                if (input.peek() == '"') { input.get(); field += '"'; }
                else { quoted = false; afterQuote = true; }
            } else {
                if (ch == '\n') ++line;
                field += ch;
            }
            continue;
        }
        if (ch == ',' || ch == '\n' || ch == '\r') {
            fields.push_back(field);
            field.clear();
            afterQuote = false;
            if (ch == ',') continue;
            if (ch == '\r' && input.peek() == '\n') input.get();
            ++line;
            return true;
        }
        if (afterQuote) {
            throw std::runtime_error(path + ": unexpected character after closing quote near line " + std::to_string(line));
        }
        if (ch == '"') {
            if (!field.empty()) throw std::runtime_error(path + ": quote inside an unquoted field near line " + std::to_string(line));
            quoted = true;
        } else field += ch;
    }
}

inline bool boolean(const std::string &value) {
    std::string normalized = trim(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (normalized == "true" || normalized == "1") return true;
    if (normalized == "false" || normalized == "0") return false;
    throw std::runtime_error("invalid boolean '" + value + "'");
}

inline long long integer(const std::string &value) {
    const std::string normalized = trim(value);
    if (normalized.empty()) throw std::runtime_error("missing integer");
    char *end = nullptr;
    errno = 0;
    const long long result = std::strtoll(normalized.c_str(), &end, 10);
    if (errno == ERANGE || end != normalized.c_str() + normalized.size())
        throw std::runtime_error("invalid integer '" + value + "'");
    return result;
}

inline int smallInteger(const std::string &value) {
    const long long result = integer(value);
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
        throw std::runtime_error("integer outside int range '" + value + "'");
    return static_cast<int>(result);
}

inline double finiteNumber(const std::string &value) {
    const std::string normalized = trim(value);
    if (normalized.empty()) throw std::runtime_error("missing number");
    char *end = nullptr;
    errno = 0;
    const double result = std::strtod(normalized.c_str(), &end);
    if (errno == ERANGE || end != normalized.c_str() + normalized.size() || !std::isfinite(result))
        throw std::runtime_error("invalid or nonfinite number '" + value + "'");
    return result;
}

} // namespace csv_detail

// Only valid, complete rows eligible for ranking become plot candidates.
// Ineligible rows may contain nan or missing numerical values and are skipped.
// Column order is immaterial; additional CSV columns are allowed.
inline std::vector<CsvRow> loadMetrics(const std::string &path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open metrics CSV: " + path);
    std::vector<std::string> fields;
    std::size_t line = 1;
    if (!csv_detail::readRecord(input, fields, line, path))
        throw std::runtime_error(path + ": empty metrics CSV");
    if (!fields.empty() && fields[0].compare(0, 3, "\xEF\xBB\xBF") == 0) fields[0].erase(0, 3);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const std::string name = csv_detail::trim(fields[i]);
        if (name.empty() || !columns.emplace(name, i).second)
            throw std::runtime_error(path + ": empty or duplicate CSV column '" + name + "'");
    }
    const char *required[] = {"model", "run_id", "epoch", "energy", "method", "readout", "region",
        "angle_bin", "low", "high", "entries", "resolution_percent", "error_percent", "valid", "complete",
        "eligible_for_ranking", "events", "event_selection_sha256", "beta_threshold",
        "expected_files", "collected_files"};
    for (const char *name : required)
        if (!columns.count(name)) throw std::runtime_error(path + ": missing required CSV column '" + name + "'");
    typedef std::tuple<std::string, std::string, int, int, std::string, std::string, std::string, int> Key;
    std::set<Key> seen;
    std::vector<CsvRow> rows;
    while (true) {
        const std::size_t rowLine = line;
        if (!csv_detail::readRecord(input, fields, line, path)) break;
        if (fields.size() == 1 && csv_detail::trim(fields[0]).empty()) continue;
        try {
            if (fields.size() != columns.size()) throw std::runtime_error("CSV field count does not match header");
            const auto get = [&](const char *name) -> const std::string & { return fields[columns.at(name)]; };
            // Validate every flag rather than short-circuiting malformed input.
            const bool valid = csv_detail::boolean(get("valid"));
            const bool complete = csv_detail::boolean(get("complete"));
            const bool eligible = csv_detail::boolean(get("eligible_for_ranking"));
            if (!valid || !complete || !eligible) continue;
            CsvRow row;
            row.model = get("model"); row.run = get("run_id");
            row.method = get("method"); row.readout = get("readout"); row.region = get("region");
            row.digest = get("event_selection_sha256");
            if (row.model.empty() || row.run.empty() || row.method.empty() || row.readout.empty() || row.region.empty() || row.digest.empty())
                throw std::runtime_error("missing comparison identity or event selection digest");
            row.epoch = csv_detail::smallInteger(get("epoch"));
            row.energy = csv_detail::smallInteger(get("energy"));
            row.angleBin = csv_detail::smallInteger(get("angle_bin"));
            row.events = csv_detail::integer(get("events"));
            row.entries = csv_detail::integer(get("entries"));
            row.expectedFiles = csv_detail::integer(get("expected_files"));
            row.collectedFiles = csv_detail::integer(get("collected_files"));
            row.low = csv_detail::finiteNumber(get("low"));
            row.high = csv_detail::finiteNumber(get("high"));
            row.resolution = csv_detail::finiteNumber(get("resolution_percent"));
            row.error = csv_detail::finiteNumber(get("error_percent"));
            row.betaThreshold = csv_detail::finiteNumber(get("beta_threshold"));
            if (row.epoch < 0 || row.energy <= 0 || row.events <= 0 || row.entries <= 0)
                throw std::runtime_error("epoch must be nonnegative; energy, events and entries must be positive");
            if (row.resolution < 0 || row.error < 0 || row.low >= row.high)
                throw std::runtime_error("JER/error must be nonnegative and angular low must be below high");
            if (row.expectedFiles <= 0 || row.collectedFiles != row.expectedFiles || row.entries > row.events)
                throw std::runtime_error("inconsistent complete input coverage or event count");
            if (row.betaThreshold < 0 || row.betaThreshold > 1 || row.low < 0 || row.high > 1)
                throw std::runtime_error("beta threshold and angular bounds must be within [0, 1]");
            const Key key(row.model, row.run, row.energy, row.epoch, row.method, row.readout, row.region, row.angleBin);
            if (!seen.insert(key).second) throw std::runtime_error("duplicate epoch/energy/method/readout/region/angle_bin group");
            rows.push_back(row);
        } catch (const std::exception &error) {
            throw std::runtime_error(path + ": CSV line " + std::to_string(rowLine) + ": " + error.what());
        }
    }
    return rows;
}

} // namespace jer_epoch_plot
#endif
