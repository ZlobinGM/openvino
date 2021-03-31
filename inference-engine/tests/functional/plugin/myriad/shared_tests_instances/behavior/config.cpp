// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/vpu_plugin_config.hpp"
#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/optional.hpp"
#include "behavior/config.hpp"

#include "myriad_devices.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace {

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine::PluginConfigParams;

const std::vector<InferenceEngine::Precision>& getPrecisions() {
    static const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };
    return precisions;
}

std::vector<std::map<std::string, std::string>> getCorrectConfigs() {
    std::vector<std::map<std::string, std::string>> correctConfigs = {
        {{KEY_LOG_LEVEL, LOG_NONE}},
        {{KEY_LOG_LEVEL, LOG_ERROR}},
        {{KEY_LOG_LEVEL, LOG_WARNING}},
        {{KEY_LOG_LEVEL, LOG_INFO}},
        {{KEY_LOG_LEVEL, LOG_DEBUG}},
        {{KEY_LOG_LEVEL, LOG_TRACE}},

        {{InferenceEngine::MYRIAD_COPY_OPTIMIZATION, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_COPY_OPTIMIZATION, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0"}},
        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"}},

        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"}},
        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "2"}},
        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "3"}},

        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_FULL}},
        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_INFER}},
        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE}},
        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE_SHAVES}},
        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE_NCES}},

        {{InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_HW_BLACK_LIST, "deconv"}},
        {{InferenceEngine::MYRIAD_HW_BLACK_LIST, "conv,pool"}},

        {{InferenceEngine::MYRIAD_WATCHDOG, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_WATCHDOG, CONFIG_VALUE(NO)}},

        // Deprecated
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_NONE}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_ERROR}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_WARNING}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_INFO}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_DEBUG}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_TRACE}},

        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}},
        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)}},

        {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}},
        {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)}},

        {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)}},
        {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO)}},

        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},

        {
            {KEY_LOG_LEVEL, LOG_INFO},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_INFER},
            {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_HW_BLACK_LIST, "deconv"},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_WATCHDOG, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(YES)},
        }
    };

    MyriadDevicesInfo info;
    if (info.getAmountOfDevices(ncDeviceProtocol_t::NC_PCIE) > 0) {
        correctConfigs.emplace_back(std::map<std::string, std::string>{{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}});
        correctConfigs.emplace_back(std::map<std::string, std::string>{{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}});
    }

    if (info.getAmountOfDevices(ncDeviceProtocol_t::NC_USB) > 0) {
        correctConfigs.emplace_back(std::map<std::string, std::string>{{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}});
        correctConfigs.emplace_back(std::map<std::string, std::string>{{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}});
    }

    return correctConfigs;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getCorrectConfigs())),
    CorrectConfigTests::getTestCaseName);

const std::vector<std::map<std::string, std::string>>& getCorrectMultiConfigs() {
    static const std::vector<std::map<std::string, std::string>> correctMultiConfigs = {
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_LOG_LEVEL, LOG_DEBUG}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, YES}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, YES}
        },

        // Deprecated
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(LOG_LEVEL), LOG_DEBUG}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), YES}
        },
    };
    return correctMultiConfigs;
}

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getCorrectMultiConfigs())),
    CorrectConfigTests::getTestCaseName);

const std::vector<std::pair<std::string, InferenceEngine::Parameter>>& getDefaultEntries() {
    static const std::vector<std::pair<std::string, InferenceEngine::Parameter>> defaultEntries = {
        {KEY_LOG_LEVEL, {LOG_NONE}},
        {InferenceEngine::MYRIAD_PROTOCOL, {std::string()}},
        {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, {true}},
        {InferenceEngine::MYRIAD_POWER_MANAGEMENT, {InferenceEngine::MYRIAD_POWER_FULL}},
        {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, {true}},
        {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, {false}},
        {InferenceEngine::MYRIAD_HW_BLACK_LIST, {std::string()}},
        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB_AUTO}},
        {InferenceEngine::MYRIAD_WATCHDOG, {std::chrono::milliseconds(1000)}},
        {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, {false}},
    };
    return defaultEntries;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectSingleOptionDefaultValueConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getDefaultEntries())));

const std::vector<std::tuple<std::string, std::string, InferenceEngine::Parameter>>& getCustomEntries() {
    static const std::vector<std::tuple<std::string, std::string, InferenceEngine::Parameter>> customEntries = {
        {KEY_LOG_LEVEL, LOG_NONE,    {LOG_NONE}},
        {KEY_LOG_LEVEL, LOG_ERROR,   {LOG_ERROR}},
        {KEY_LOG_LEVEL, LOG_WARNING, {LOG_WARNING}},
        {KEY_LOG_LEVEL, LOG_INFO,    {LOG_INFO}},
        {KEY_LOG_LEVEL, LOG_DEBUG,   {LOG_DEBUG}},
        {KEY_LOG_LEVEL, LOG_TRACE,   {LOG_TRACE}},

        {VPU_CONFIG_KEY(LOG_LEVEL), LOG_NONE,    {LOG_NONE}},
        {VPU_CONFIG_KEY(LOG_LEVEL), LOG_ERROR,   {LOG_ERROR}},
        {VPU_CONFIG_KEY(LOG_LEVEL), LOG_WARNING, {LOG_WARNING}},
        {VPU_CONFIG_KEY(LOG_LEVEL), LOG_INFO,    {LOG_INFO}},
        {VPU_CONFIG_KEY(LOG_LEVEL), LOG_DEBUG,   {LOG_DEBUG}},
        {VPU_CONFIG_KEY(LOG_LEVEL), LOG_TRACE,   {LOG_TRACE}},

        {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB, {InferenceEngine::MYRIAD_USB}},
        {InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE, {InferenceEngine::MYRIAD_PCIE}},

        {VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB), {VPU_MYRIAD_CONFIG_VALUE(USB)}},
        {VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE), {VPU_MYRIAD_CONFIG_VALUE(PCIE)}},

        {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_FULL,          {InferenceEngine::MYRIAD_POWER_FULL}},
        {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_INFER,         {InferenceEngine::MYRIAD_POWER_INFER}},
        {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE,         {InferenceEngine::MYRIAD_POWER_STAGE}},
        {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE_SHAVES,  {InferenceEngine::MYRIAD_POWER_STAGE_SHAVES}},
        {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE_NCES,    {InferenceEngine::MYRIAD_POWER_STAGE_NCES}},

        {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, InferenceEngine::PluginConfigParams::NO, {false}},

        {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES), {true}},
        {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO), {false}},

        {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_HW_BLACK_LIST, "deconv", {"deconv"}},
        {InferenceEngine::MYRIAD_HW_BLACK_LIST, "conv,pool",   {"conv,pool"}},

        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0", {"0"}},
        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "1", {"1"}},
        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10", {"10"}},

        {InferenceEngine::MYRIAD_WATCHDOG, InferenceEngine::PluginConfigParams::YES, {std::chrono::milliseconds(1000)}},
        {InferenceEngine::MYRIAD_WATCHDOG, InferenceEngine::PluginConfigParams::NO, {std::chrono::milliseconds(0)}},

        {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, InferenceEngine::PluginConfigParams::NO, {false}},

        {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES), {true}},
        {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO), {false}},
    };
    return customEntries;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectSingleOptionCustomValueConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getCustomEntries())));

const std::vector<std::string>& getPublicOptions() {
    static const std::vector<std::string> publicOptions = {
        KEY_LOG_LEVEL,
        VPU_CONFIG_KEY(LOG_LEVEL),
        InferenceEngine::MYRIAD_PROTOCOL,
        VPU_MYRIAD_CONFIG_KEY(PROTOCOL),
        InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION,
        VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
        InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME,
        VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME)
    };
    return publicOptions;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigPublicOptionsTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getPublicOptions())));

const std::vector<std::string>& getPrivateOptions() {
    static const std::vector<std::string> privateOptions = {
        InferenceEngine::MYRIAD_COPY_OPTIMIZATION,
        InferenceEngine::MYRIAD_POWER_MANAGEMENT,
        InferenceEngine::MYRIAD_HW_EXTRA_SPLIT,
        InferenceEngine::MYRIAD_HW_BLACK_LIST,
        InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB,
        InferenceEngine::MYRIAD_WATCHDOG
    };
    return privateOptions;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigPrivateOptionsTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getPrivateOptions())));

const std::vector<std::map<std::string, std::string>>& getIncorrectConfigs() {
    static const std::vector<std::map<std::string, std::string>> incorrectConfigs = {
        {{KEY_LOG_LEVEL, "INCORRECT_LOG_LEVEL"}},

        {{InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "ON"}},
        {{InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "OFF"}},

        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, "FULL"}},
        {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, "ECONOM"}},

        {{InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"}},
        {{InferenceEngine::MYRIAD_PROTOCOL, "LAN"}},

        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "OFF"}},

        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "OFF"}},

        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-1"}},
        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-10"}},

        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "OFF"}},

        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "Two"}},
        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "SINGLE"}},

        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "OFF"}},

        {{InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, "ON"}},
        {{InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, "OFF"}},

        {{InferenceEngine::MYRIAD_WATCHDOG, "ON"}},
        {{InferenceEngine::MYRIAD_WATCHDOG, "OFF"}},

        // Deprecated
        {{VPU_CONFIG_KEY(LOG_LEVEL), "INCORRECT_LOG_LEVEL"}},

        {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "BLUETOOTH"}},
        {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "LAN"}},

        {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"}},
        {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "OFF"}},

        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "ON"}},
        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "OFF"}},

        {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "ON"}},
        {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "OFF"}},

        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}},

        {
            {KEY_LOG_LEVEL, LOG_INFO},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "ON"},
            {InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"},
            {InferenceEngine::MYRIAD_POWER_MANAGEMENT, "FULL"},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, "ON"},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"},
            {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-10"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "OFF"},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "ON"},
            {InferenceEngine::MYRIAD_WATCHDOG, "OFF"},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"},
        }
    };
    return incorrectConfigs;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getIncorrectConfigs())),
    IncorrectConfigTests::getTestCaseName);

const std::vector<std::map<std::string, std::string>>& getIncorrectMultiConfigs() {
    static const std::vector<std::map<std::string, std::string>> incorrectMultiConfigs = {
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_LOG_LEVEL, "INCORRECT_LOG_LEVEL"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"}
        },

        // Deprecated
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(LOG_LEVEL), "INCORRECT_LOG_LEVEL"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "BLUETOOTH"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "OFF"}
        },
    };
    return incorrectMultiConfigs;
}

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getIncorrectMultiConfigs())),
    IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigSingleOptionTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values("INCORRECT_KEY")));

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(std::map<std::string, std::string>{})),
    CorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getCorrectMultiConfigs())),
    CorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(std::map<std::string, std::string>{{"INCORRECT_KEY", "INCORRECT_VALUE"}})),
    IncorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getIncorrectMultiConfigs())),
    IncorrectConfigAPITests::getTestCaseName);

const std::vector<std::tuple<
    std::tuple<std::string, std::string, InferenceEngine::Parameter>,
    std::tuple<std::string, std::string, InferenceEngine::Parameter>>>& getCustomAndEnvironmentEntries() {
    static const std::vector<std::tuple<
        std::tuple<std::string, std::string, InferenceEngine::Parameter>,
        std::tuple<std::string, std::string, InferenceEngine::Parameter>>> customAndEnvironmentEntries = {
        {{KEY_LOG_LEVEL, LOG_NONE, {LOG_NONE}},
         {"IE_VPU_LOG_LEVEL", LOG_ERROR, {LOG_ERROR}}},

        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0", {"0"}},
         {"IE_VPU_TILING_CMX_LIMIT_KB", "1", {"1"}}},

        {{InferenceEngine::MYRIAD_WATCHDOG, InferenceEngine::PluginConfigParams::YES, {std::chrono::milliseconds(1000)}},
         {"IE_VPU_MYRIAD_WATCHDOG_INTERVAL", "100", {std::chrono::milliseconds(100)}}},

        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_WARNING, {LOG_WARNING}},
         {"IE_VPU_LOG_LEVEL", LOG_NONE, {LOG_NONE}}},
    };
    return customAndEnvironmentEntries;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectSingleOptionCustomAndEnvironmentValueConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getCustomAndEnvironmentEntries())));
} // namespace
