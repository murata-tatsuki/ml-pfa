#ifndef JER_TRAINING_LOSS_H
#define JER_TRAINING_LOSS_H

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jer_epoch_plot {

struct LossEpoch {
    int epoch = 0;
    double trainTotal = std::numeric_limits<double>::quiet_NaN();
    double validationTotal = std::numeric_limits<double>::quiet_NaN();
    double learningRate = std::numeric_limits<double>::quiet_NaN();
    std::map<std::string, double> train, validation;
};

struct LossHistory {
    std::string source;
    std::map<int, LossEpoch> epochs;
    int betaStart = -1, energyStart = -1;
    std::string activationSource;
    std::vector<std::string> warnings;
};

// Activation schedules are not inferred from component values: a term may be
// printed while its coefficient is zero. Supply verified run settings instead.
inline void setLossActivation(LossHistory &history, int betaStart, int energyStart,
                              const std::string &source) {
    if (betaStart < -1 || energyStart < -1)
        throw std::runtime_error("loss activation epochs must be >= 0, or -1 for unknown");
    if ((betaStart >= 0 || energyStart >= 0) && source.empty())
        throw std::runtime_error("loss activation settings require a source description");
    history.betaStart = betaStart;
    history.energyStart = energyStart;
    history.activationSource = source;
}

namespace training_loss_detail {

inline std::string trim(const std::string &value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return std::string();
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

inline bool number(const std::string &value, double &result) {
    const std::string clean = trim(value);
    if (clean.empty()) return false;
    char *end = nullptr;
    errno = 0;
    result = std::strtod(clean.c_str(), &end);
    return end == clean.c_str() + clean.size() && end != clean.c_str() &&
           errno != ERANGE && std::isfinite(result);
}

inline std::string firstToken(const std::string &value) {
    const std::string clean = trim(value);
    return clean.substr(0, clean.find_first_of(" \t\r\n("));
}

struct Block {
    LossEpoch value;
    std::size_t line = 0;
    int mode = 0; // 0: unknown, 1: train, 2: validation.
    bool present = false, invalid = false, trainTotal = false, validationTotal = false;
};

inline std::string location(const std::string &path, std::size_t line, int epoch) {
    return path + ":" + std::to_string(line) + " (epoch " + std::to_string(epoch) + ")";
}

inline void flush(Block &block, LossHistory &history, const std::set<int> &duplicates) {
    if (!block.present) return;
    if (!block.trainTotal || !block.validationTotal || block.invalid) {
        history.warnings.push_back(location(history.source, block.line, block.value.epoch) +
            ": skipped incomplete or invalid epoch; require one finite train total and validation total");
    } else if (!duplicates.count(block.value.epoch)) {
        history.epochs.emplace(block.value.epoch, block.value);
    }
}

} // namespace training_loss_detail

// Reads epoch summaries from a single training process log (normally rank0).
// Batch progress such as "epoch=... train_batch=... loss=..." is ignored.
// A complete epoch needs both printed totals. Missing/nonfinite components stay
// absent, never zero. Duplicate epoch boundaries invalidate *all* occurrences.
// Warnings are returned for the caller to display or record with the plots.
inline LossHistory readTrainingLoss(const std::string &path) {
    using namespace training_loss_detail;
    std::ifstream input(path.c_str());
    if (!input) throw std::runtime_error("Cannot open training loss log: " + path);

    const std::regex epochPattern("^Training[ \\t]+epoch[ \\t]+([0-9]+)[ \\t]*$");
    const std::regex componentPattern("^(?:(train|test|validation)[ \\t]+)?(L_[A-Za-z0-9_]+)[ \\t]*=[ \\t]*(.*)$");
    const std::regex trainTotalPattern("^train[ \\t]+loss[ \\t]*:[ \\t]*(.*)$");
    const std::regex validationTotalPattern("^Returning[ \\t]+(.*)$");
    const std::regex learningRatePattern("^learning[ \\t]+rate[ \\t]*:[ \\t]*(.*)$");

    LossHistory history;
    history.source = path;
    Block block;
    std::set<int> seen, duplicates;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(line);
        std::smatch match;
        if (std::regex_match(line, match, epochPattern)) {
            flush(block, history, duplicates);
            block = Block();
            long long epoch = -1;
            try { epoch = std::stoll(match[1].str()); }
            catch (const std::exception &) {}
            if (epoch < 0 || epoch > std::numeric_limits<int>::max())
                throw std::runtime_error(path + ":" + std::to_string(lineNumber) + ": invalid epoch index");
            block.present = true;
            block.line = lineNumber;
            block.value.epoch = static_cast<int>(epoch);
            if (!seen.insert(block.value.epoch).second) {
                duplicates.insert(block.value.epoch);
                history.epochs.erase(block.value.epoch);
                history.warnings.push_back(location(path, lineNumber, block.value.epoch) +
                    ": duplicate epoch boundary; all occurrences excluded to prevent an ambiguous checkpoint join");
            }
            continue;
        }
        if (!block.present) continue;
        const std::string where = location(path, lineNumber, block.value.epoch);
        if (std::regex_match(line, match, trainTotalPattern)) {
            double value = 0;
            if (block.trainTotal || block.validationTotal || !number(trim(match[1].str()), value)) {
                block.invalid = true;
                history.warnings.push_back(where + ": duplicate, out-of-order or nonfinite train total");
            } else {
                block.value.trainTotal = value;
                block.trainTotal = true;
            }
            continue;
        }
        if (std::regex_match(line, match, validationTotalPattern)) {
            double value = 0;
            if (!block.trainTotal || block.validationTotal || !number(trim(match[1].str()), value)) {
                block.invalid = true;
                history.warnings.push_back(where + ": duplicate, out-of-order or nonfinite validation total");
            } else {
                block.value.validationTotal = value;
                block.validationTotal = true;
            }
            continue;
        }
        // Summaries end with Returning. Do not absorb metrics printed between
        // completed summaries and the next Training epoch boundary.
        if (block.validationTotal) continue;
        if (std::regex_match(line, match, learningRatePattern)) {
            std::string values = trim(match[1].str());
            if (!values.empty() && values.front() == '[' && values.back() == ']') {
                values = trim(values.substr(1, values.size() - 2));
                // The logger prints its cumulative LR history, not just this
                // epoch's value. The newest rate is the final list element.
                const std::size_t comma = values.find_last_of(',');
                if (comma != std::string::npos) values = trim(values.substr(comma + 1));
            }
            double value = 0;
            if (number(values, value)) block.value.learningRate = value;
            else history.warnings.push_back(where + ": missing or nonfinite learning rate; left unavailable");
            continue;
        }
        if (std::regex_match(line, match, componentPattern)) {
            const std::string prefix = match[1].str(), name = match[2].str();
            if (!prefix.empty()) block.mode = prefix == "train" ? 1 : 2;
            if (!block.mode) {
                history.warnings.push_back(where + ": component " + name + " before train/test section; skipped");
                continue;
            }
            if (block.mode == 1 && block.trainTotal) {
                block.invalid = true;
                history.warnings.push_back(where + ": training component after train total; epoch excluded");
                continue;
            }
            double value = 0;
            if (!number(firstToken(match[3].str()), value)) {
                history.warnings.push_back(where + ": nonfinite or invalid component " + name + "; skipped");
                continue;
            }
            std::map<std::string, double> &components = block.mode == 1 ? block.value.train : block.value.validation;
            if (!components.emplace(name, value).second) {
                block.invalid = true;
                history.warnings.push_back(where + ": duplicate component " + name + "; epoch excluded");
            }
        }
    }
    if (input.bad()) throw std::runtime_error("Failed reading training loss log: " + path);
    flush(block, history, duplicates);
    if (history.epochs.empty())
        throw std::runtime_error("No complete unambiguous epochs in training loss log: " + path);
    return history;
}

} // namespace jer_epoch_plot

#endif
