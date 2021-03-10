// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <unordered_set>
#include <chrono>

#include <vpu/parsed_config.hpp>

#include <mvnc.h>

namespace vpu {
namespace MyriadPlugin {

// Must be synchronized with firmware side.
VPU_DECLARE_ENUM(PowerConfig,
    FULL         = 0,
    INFER        = 1,
    STAGE        = 2,
    STAGE_SHAVES = 3,
    STAGE_NCES   = 4,
)

// Must be synchronized with firmware side.
VPU_DECLARE_ENUM(MovidiusDdrType,
    AUTO        = 0,
    MICRON_2GB  = 1,
    SAMSUNG_2GB = 2,
    HYNIX_2GB   = 3,
    MICRON_1GB  = 4,
)

class MyriadConfig : public virtual ParsedConfig {
public:
    const std::string& pluginLogFilePath() const {
        return _pluginLogFilePath;
    }

    bool forceReset() const {
        return _forceReset;
    }

    bool asyncDma() const {
        return _enableAsyncDma;
    }

    PowerConfig powerConfig() const {
        return _powerConfig;
    }

    ncDevicePlatform_t platform() const {
        return _platform;
    }

    const std::chrono::milliseconds& watchdogInterval() const {
        return _watchdogInterval;
    }

    const std::chrono::seconds& deviceConnectTimeout() const {
        return _deviceConnectTimeout;
    }

    const std::string& deviceName() const {
        return _deviceName;
    }

    MovidiusDdrType memoryType() const {
        return _memoryType;
    }

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    const std::unordered_set<std::string>& getDeprecatedOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    std::string _pluginLogFilePath;
    bool _forceReset = false;
    bool _enableAsyncDma = true;
    PowerConfig _powerConfig = PowerConfig::FULL;
    ncDevicePlatform_t _platform = NC_ANY_PLATFORM;
    std::chrono::milliseconds _watchdogInterval = std::chrono::milliseconds(1000);
    std::chrono::seconds _deviceConnectTimeout = std::chrono::seconds(15);
    std::string _deviceName;
    MovidiusDdrType _memoryType = MovidiusDdrType::AUTO;
};

}  // namespace MyriadPlugin
}  // namespace vpu
