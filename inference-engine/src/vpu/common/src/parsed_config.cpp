// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/parsed_config.hpp>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <memory>
#include <map>

#include <debug.h>
#include <ie_plugin_config.hpp>

#include <vpu/utils/string.hpp>

namespace vpu {

const std::unordered_set<std::string>& ParsedConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfigBase::getCompileOptions(), {
        //
        // Private options
        //

        ie::MYRIAD_COPY_OPTIMIZATION,
        ie::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL,
        ie::MYRIAD_ENABLE_EARLY_ELTWISE_RELU_FUSION,
        ie::MYRIAD_ENABLE_CUSTOM_RESHAPE_PARAM,

        //
        // Debug options
        //

        ie::MYRIAD_NONE_LAYERS,
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& ParsedConfig::getRunTimeOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = ParsedConfigBase::getRunTimeOptions();
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& ParsedConfig::getDeprecatedOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = ParsedConfigBase::getDeprecatedOptions();
IE_SUPPRESS_DEPRECATED_END

    return options;
}

void ParsedConfig::parse(const std::map<std::string, std::string>& config) {
    static const auto parseStrides = [](const std::string& src) {
        auto configStrides = src;
        configStrides.pop_back();

        const auto inputs = ie::details::split(configStrides, "],");

        std::map<std::string, std::vector<int>> stridesMap;

        for (const auto& input : inputs) {
            std::vector<int> strides;

            const auto pair = ie::details::split(input, "[");
            IE_ASSERT(pair.size() == 2)
                    << "Invalid config value \"" << input << "\" "
                    << "for VPU_TENSOR_STRIDES, does not match the pattern: tensor_name[strides]";

            const auto strideValues = ie::details::split(pair.at(1), ",");

            for (const auto& stride : strideValues) {
                strides.insert(strides.begin(), std::stoi(stride));
            }

            stridesMap.insert({pair.at(0), strides});
        }

        return stridesMap;
    };

    const auto parseStringSet = [](const std::string& value) {
        return splitStringList<ie::details::caseless_set<std::string>>(value, ',');
    };

    ParsedConfigBase::parse(config);

    setOption(_compileConfig.checkPreprocessingInsideModel,  switches, config, ie::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL);
    setOption(_compileConfig.enableEarlyEltwiseReLUFusion,   switches, config, ie::MYRIAD_ENABLE_EARLY_ELTWISE_RELU_FUSION);
    setOption(_compileConfig.enableCustomReshapeParam,       switches, config, ie::MYRIAD_ENABLE_CUSTOM_RESHAPE_PARAM);

    setOption(_compileConfig.noneLayers,                               config, ie::MYRIAD_NONE_LAYERS, parseStringSet);

    auto isPositive = [](int value) {
        return value >= 0;
    };

    auto isDefaultValue = [](int value) {
        return value == -1;
    };

    auto preprocessCompileOption = [&](const std::string& src) {
        int value = parseInt(src);

        if (isPositive(value) || isDefaultValue(value)) {
            return value;
        }

        throw std::invalid_argument("Value must be positive or default(-1).");
    };
}

}  // namespace vpu
