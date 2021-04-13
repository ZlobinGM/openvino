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

        {{InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_HW_BLACK_LIST, "deconv"}},
        {{InferenceEngine::MYRIAD_HW_BLACK_LIST, "conv,pool"}},

        {{InferenceEngine::MYRIAD_HW_INJECT_STAGES, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_HW_INJECT_STAGES, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_HW_DILATION, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_HW_DILATION, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_WATCHDOG, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_WATCHDOG, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_LAYER}},
        {{InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE}},

        {{KEY_PERF_COUNT, CONFIG_VALUE(YES)}},
        {{KEY_PERF_COUNT, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, CONFIG_VALUE(NO)}},

        {
            {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "2"},
            {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "2"},
        },

        {{InferenceEngine::MYRIAD_TENSOR_STRIDES, "tensor[1,2,3,4]"}},

        {{InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, CONFIG_VALUE(NO)}},

        {{KEY_EXCLUSIVE_ASYNC_REQUESTS, CONFIG_VALUE(YES)}},
        {{KEY_EXCLUSIVE_ASYNC_REQUESTS, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_DUMP_ALL_PASSES, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_DUMP_ALL_PASSES, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_DISABLE_REORDER, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_DISABLE_REORDER, CONFIG_VALUE(NO)}},

        {{KEY_DEVICE_ID, ""}},

        {{InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "10"}},
        {{InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "15"}},
        {{InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "20"}},

        {{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_CUSTOM_LAYERS, ""}},

        {{KEY_CONFIG_FILE, ""}},

        {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO}},

        {{InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, CONFIG_VALUE(NO)}},

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

        {{VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), CONFIG_VALUE(YES)}},
        {{VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), CONFIG_VALUE(NO)}},

        {{VPU_CONFIG_KEY(CUSTOM_LAYERS), ""}},

        {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}},

        {
            {KEY_LOG_LEVEL, LOG_INFO},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_INFER},
            {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_HW_BLACK_LIST, "deconv"},
            {InferenceEngine::MYRIAD_HW_INJECT_STAGES, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_HW_DILATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_WATCHDOG, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "10"},
            {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "10"},
            {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_LAYER},
            {KEY_PERF_COUNT, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_TENSOR_STRIDES, "tensor[1,2,3,4]"},
            {InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, CONFIG_VALUE(NO)},
            {KEY_EXCLUSIVE_ASYNC_REQUESTS, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_DUMP_ALL_PASSES, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_DISABLE_REORDER, CONFIG_VALUE(NO)},
            {KEY_DEVICE_ID, ""},
            {InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "10"},
            {InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)},
            {InferenceEngine::MYRIAD_CUSTOM_LAYERS, ""},
            {KEY_CONFIG_FILE, ""},
            {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO},
            {InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, CONFIG_VALUE(NO)},
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
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_PERF_COUNT, YES}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_EXCLUSIVE_ASYNC_REQUESTS, YES}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_DEVICE_ID, ""}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_CUSTOM_LAYERS, ""}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_CONFIG_FILE, ""}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, YES}
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
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(CUSTOM_LAYERS), ""}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}
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
        {InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, {true}},
        {InferenceEngine::MYRIAD_HW_INJECT_STAGES, {InferenceEngine::MYRIAD_HW_INJECT_STAGES_AUTO}},
        {InferenceEngine::MYRIAD_HW_DILATION, {false}},
        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB_AUTO}},
        {InferenceEngine::MYRIAD_WATCHDOG, {std::chrono::milliseconds(1000)}},
        {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, {false}},
        {InferenceEngine::MYRIAD_PERF_REPORT_MODE, {InferenceEngine::MYRIAD_PER_LAYER}},
        {KEY_PERF_COUNT, {false}},
        {InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, {true}},
        {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES_AUTO}},
        {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS_AUTO}},
        {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES_AUTO}},
        {InferenceEngine::MYRIAD_IR_WITH_SCALES_DIRECTORY, {std::string()}},
        {InferenceEngine::MYRIAD_TENSOR_STRIDES, {std::map<std::string, std::vector<int>>()}},
        {InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, {false}},
        {InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, {false}},
        {InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, {false}},
        {KEY_EXCLUSIVE_ASYNC_REQUESTS, {false}},
        {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, {true}},
        {InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, {false}},
        {InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, {true}},
        {InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, {false}},
        {InferenceEngine::MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME, {std::string()}},
        {InferenceEngine::MYRIAD_DUMP_ALL_PASSES_DIRECTORY, {std::string()}},
        {InferenceEngine::MYRIAD_DUMP_ALL_PASSES, {false}},
        {InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, {false}},
        {InferenceEngine::MYRIAD_DISABLE_REORDER, {false}},
        {KEY_DEVICE_ID, {std::string()}},
        {InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, {std::chrono::seconds(15)}},
        {InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, {true}},
        {InferenceEngine::MYRIAD_CUSTOM_LAYERS, {std::string()}},
        {KEY_CONFIG_FILE, {std::string()}},
        {InferenceEngine::MYRIAD_DDR_TYPE, {InferenceEngine::MYRIAD_DDR_AUTO}},
        {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, {false}},
        {VPU_MYRIAD_CONFIG_KEY(PLATFORM), {std::string()}},
        {InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, {true}},
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

        {VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480), {VPU_MYRIAD_CONFIG_VALUE(2480)}},

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

        {InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_HW_BLACK_LIST, "deconv", {"deconv"}},
        {InferenceEngine::MYRIAD_HW_BLACK_LIST, "conv,pool",   {"conv,pool"}},

        {InferenceEngine::MYRIAD_HW_DILATION, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_HW_DILATION, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_HW_INJECT_STAGES, InferenceEngine::PluginConfigParams::YES, {InferenceEngine::PluginConfigParams::YES}},
        {InferenceEngine::MYRIAD_HW_INJECT_STAGES, InferenceEngine::PluginConfigParams::NO, {InferenceEngine::PluginConfigParams::NO}},

        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0", {"0"}},
        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "1", {"1"}},
        {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10", {"10"}},

        {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "0", {"0"}},
        {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "1", {"1"}},
        {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "10", {"10"}},

        {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "0", {"0"}},
        {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "1", {"1"}},
        {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "10", {"10"}},

        {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1", {"1"}},
        {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "2", {"2"}},
        {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "3", {"3"}},

        {InferenceEngine::MYRIAD_WATCHDOG, InferenceEngine::PluginConfigParams::YES, {std::chrono::milliseconds(1000)}},
        {InferenceEngine::MYRIAD_WATCHDOG, InferenceEngine::PluginConfigParams::NO, {std::chrono::milliseconds(0)}},

        {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, InferenceEngine::PluginConfigParams::NO, {false}},

        {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES), {true}},
        {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO), {false}},

        {InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_LAYER, {InferenceEngine::MYRIAD_PER_LAYER}},
        {InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE, {InferenceEngine::MYRIAD_PER_STAGE}},

        {KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES, {true}},
        {KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_IR_WITH_SCALES_DIRECTORY, "/", {std::string("/")}},

        {InferenceEngine::MYRIAD_TENSOR_STRIDES, "tensor[1,2,3,4]", {std::map<std::string, std::vector<int>>{{"tensor", {4, 3, 2, 1}}}}},

        {InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, InferenceEngine::PluginConfigParams::NO, {false}},

        {KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES, {true}},
        {KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME, "/", {std::string("/")}},

        {InferenceEngine::MYRIAD_DUMP_ALL_PASSES_DIRECTORY, "/", {std::string("/")}},

        {InferenceEngine::MYRIAD_DUMP_ALL_PASSES, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_DUMP_ALL_PASSES, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_DISABLE_REORDER, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_DISABLE_REORDER, InferenceEngine::PluginConfigParams::NO, {false}},

        {InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "10", {std::chrono::seconds(10)}},
        {InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "20", {std::chrono::seconds(20)}},

        {InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, InferenceEngine::PluginConfigParams::NO, {false}},

        {VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), CONFIG_VALUE(YES), {true}},
        {VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), CONFIG_VALUE(NO), {false}},

        {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO,        {InferenceEngine::MYRIAD_DDR_AUTO}},
        {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_MICRON_2GB,  {InferenceEngine::MYRIAD_DDR_MICRON_2GB}},
        {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_SAMSUNG_2GB, {InferenceEngine::MYRIAD_DDR_SAMSUNG_2GB}},
        {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_HYNIX_2GB,   {InferenceEngine::MYRIAD_DDR_HYNIX_2GB}},
        {InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_MICRON_1GB,  {InferenceEngine::MYRIAD_DDR_MICRON_1GB}},

        {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO),    {VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}},
        {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB),  {VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB)}},
        {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB), {VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB)}},
        {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB),   {VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB)}},
        {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB),  {VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB)}},

        {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, InferenceEngine::PluginConfigParams::NO, {false}},

        {VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES), {true}},
        {VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO), {false}},

        {InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, InferenceEngine::PluginConfigParams::YES, {true}},
        {InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, InferenceEngine::PluginConfigParams::NO, {false}},
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
        VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME),
        KEY_PERF_COUNT,
        InferenceEngine::MYRIAD_THROUGHPUT_STREAMS,
        KEY_EXCLUSIVE_ASYNC_REQUESTS,
        KEY_DEVICE_ID,
        InferenceEngine::MYRIAD_CUSTOM_LAYERS,
        VPU_CONFIG_KEY(CUSTOM_LAYERS),
        KEY_CONFIG_FILE,
        InferenceEngine::MYRIAD_DDR_TYPE,
        VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE),
        InferenceEngine::MYRIAD_ENABLE_FORCE_RESET,
        VPU_MYRIAD_CONFIG_KEY(FORCE_RESET),
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
        InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE,
        InferenceEngine::MYRIAD_HW_BLACK_LIST,
        InferenceEngine::MYRIAD_HW_INJECT_STAGES,
        InferenceEngine::MYRIAD_HW_DILATION,
        InferenceEngine::MYRIAD_NUMBER_OF_SHAVES,
        InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES,
        InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB,
        InferenceEngine::MYRIAD_WATCHDOG,
        InferenceEngine::MYRIAD_PERF_REPORT_MODE,
        InferenceEngine::MYRIAD_PACK_DATA_IN_CMX,
        InferenceEngine::MYRIAD_IR_WITH_SCALES_DIRECTORY,
        InferenceEngine::MYRIAD_TENSOR_STRIDES,
        InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS,
        InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR,
        InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING,
        InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS,
        InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU,
        InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING,
        InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION,
        InferenceEngine::MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME,
        InferenceEngine::MYRIAD_DUMP_ALL_PASSES_DIRECTORY,
        InferenceEngine::MYRIAD_DUMP_ALL_PASSES,
        InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES,
        InferenceEngine::MYRIAD_DISABLE_REORDER,
        InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT,
        InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH,
        VPU_CONFIG_KEY(DETECT_NETWORK_BATCH),
        InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL,
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

        {{InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "-1"}},
        {{InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "-10"}},

        {{InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "-1"}},
        {{InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "-10"}},

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

        {{InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, "ON"}},
        {{InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, "OFF"}},

        {{InferenceEngine::MYRIAD_HW_INJECT_STAGES, "ON"}},
        {{InferenceEngine::MYRIAD_HW_INJECT_STAGES, "OFF"}},

        {{InferenceEngine::MYRIAD_HW_DILATION, "ON"}},
        {{InferenceEngine::MYRIAD_HW_DILATION, "OFF"}},

        {{InferenceEngine::MYRIAD_WATCHDOG, "ON"}},
        {{InferenceEngine::MYRIAD_WATCHDOG, "OFF"}},

        {{InferenceEngine::MYRIAD_PERF_REPORT_MODE, "PER_LAYER"}},
        {{InferenceEngine::MYRIAD_PERF_REPORT_MODE, "STAGE"}},

        {{KEY_PERF_COUNT, "ON"}},
        {{KEY_PERF_COUNT, "OFF"}},

        {{InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, "ON"}},
        {{InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, "OFF"}},

        {{InferenceEngine::MYRIAD_TENSOR_STRIDES, "tensor(1,2,3,4)"}},

        {{InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, "ON"}},
        {{InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, "OFF"}},

        {{InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, "ON"}},
        {{InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, "OFF"}},

        {{InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, "OFF"}},

        {{KEY_EXCLUSIVE_ASYNC_REQUESTS, "ON"}},
        {{KEY_EXCLUSIVE_ASYNC_REQUESTS, "OFF"}},

        {{InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, "OFF"}},

        {{InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, "OFF"}},

        {{InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, "OFF"}},

        {{InferenceEngine::MYRIAD_DUMP_ALL_PASSES, "ON"}},
        {{InferenceEngine::MYRIAD_DUMP_ALL_PASSES, "OFF"}},

        {{InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, "ON"}},
        {{InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, "OFF"}},

        {{InferenceEngine::MYRIAD_DISABLE_REORDER, "ON"}},
        {{InferenceEngine::MYRIAD_DISABLE_REORDER, "OFF"}},

        {{InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "-1"}},
        {{InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "-10"}},

        {{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, "ON"}},
        {{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, "OFF"}},

        {{InferenceEngine::MYRIAD_DDR_TYPE, "AUTO"}},
        {{InferenceEngine::MYRIAD_DDR_TYPE, "2GB"}},
        {{InferenceEngine::MYRIAD_DDR_TYPE, "1GB"}},

        {{InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, "ON"}},
        {{InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, "OFF"}},

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

        {{VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), "ON"}},
        {{VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), "OFF"}},

        {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "AUTO"}},
        {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "2GB"}},
        {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "1GB"}},

        {
            {KEY_LOG_LEVEL, LOG_INFO},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "ON"},
            {InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"},
            {InferenceEngine::MYRIAD_POWER_MANAGEMENT, "FULL"},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_HW_EXTRA_SPLIT, "ON"},
            {InferenceEngine::MYRIAD_HW_POOL_CONV_MERGE, "ON"},
            {InferenceEngine::MYRIAD_HW_INJECT_STAGES, "ON"},
            {InferenceEngine::MYRIAD_HW_DILATION, "ON"},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"},
            {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "-10"},
            {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "-10"},
            {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-10"},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "Two"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "ON"},
            {InferenceEngine::MYRIAD_WATCHDOG, "OFF"},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"},
            {InferenceEngine::MYRIAD_PERF_REPORT_MODE, "PER_LAYER"},
            {KEY_PERF_COUNT, "ON"},
            {InferenceEngine::MYRIAD_PACK_DATA_IN_CMX, "OFF"},
            {InferenceEngine::MYRIAD_TENSOR_STRIDES, "tensor(1,2,3,4)"},
            {InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, "OFF"},
            {InferenceEngine::MYRIAD_FORCE_PURE_TENSOR_ITERATOR, "OFF"},
            {InferenceEngine::MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING, "OFF"},
            {KEY_EXCLUSIVE_ASYNC_REQUESTS, "ON"},
            {InferenceEngine::MYRIAD_ENABLE_REPL_WITH_SCRELU, "OFF"},
            {InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING, "OFF"},
            {InferenceEngine::MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION, "OFF"},
            {InferenceEngine::MYRIAD_DUMP_ALL_PASSES, "OFF"},
            {InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES, "OFF"},
            {InferenceEngine::MYRIAD_DISABLE_REORDER, "OFF"},
            {InferenceEngine::MYRIAD_DEVICE_CONNECT_TIMEOUT, "-10"},
            {InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, "OFF"},
            {InferenceEngine::MYRIAD_DDR_TYPE, "AUTO"},
            {InferenceEngine::MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL, "OFF"},
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
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_PERF_COUNT, "ON"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "ONE"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_EXCLUSIVE_ASYNC_REQUESTS, "ON"}
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_DDR_TYPE, "1GB"}
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
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "1GB"}
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

        {{InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "5", {"5"}},
         {"IE_VPU_NUMBER_OF_SHAVES_AND_CMX_SLICES", "6", {"6"}}},

        {{InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "5", {"5"}},
         {"IE_VPU_NUMBER_OF_SHAVES_AND_CMX_SLICES", "6", {"6"}}},

        {{InferenceEngine::MYRIAD_DUMP_ALL_PASSES, InferenceEngine::PluginConfigParams::YES, {true}},
         {"IE_VPU_DUMP_ALL_PASSES", "0", {false}}},

        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, InferenceEngine::PluginConfigParams::YES, {true}},
         {"IE_VPU_MYRIAD_FORCE_RESET", "0", {false}}},

        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_WARNING, {LOG_WARNING}},
         {"IE_VPU_LOG_LEVEL", LOG_NONE, {LOG_NONE}}},

        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES), {true}},
         {"IE_VPU_MYRIAD_FORCE_RESET", "0", {false}}},
    };
    return customAndEnvironmentEntries;
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectSingleOptionCustomAndEnvironmentValueConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getCustomAndEnvironmentEntries())));
} // namespace
