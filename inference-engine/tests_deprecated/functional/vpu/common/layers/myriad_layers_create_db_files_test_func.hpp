#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "weights_for_convolution_test.h"

#include "conv_ref.hpp"
using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(kernel, param_size);
PRETTY_PARAM(stride, param_size);
PRETTY_PARAM(pad, param_size);
PRETTY_PARAM(out_channels, int);
PRETTY_PARAM(group, int);
PRETTY_PARAM(dilation_factor, param_size);
PRETTY_PARAM(layoutPreference, vpu::LayoutPreference);

typedef myriadLayerTestBaseWithParam<tuple<std::vector<size_t>, std::vector<size_t>, kernel, stride, pad
        , group,dilation_factor, layoutPreference >> myriadLayerConvolution_tests;

TEST_P(myriadLayerConvolution_tests, Convolution) {
    std::vector<size_t> inputHW = get<1>(GetParam());
    std::vector<size_t> IO = get<0>(GetParam());
    size_t inputC = IO[0];
    tensor_test_params input_dims = {1, inputC, inputHW[0], inputHW[1]};
    param_size kernel = get<2>(GetParam());
    param_size stride = get<3>(GetParam());
    param_size pad = get<4>(GetParam());
    size_t out_channels = IO[1];
    size_t group = get<5>(GetParam());
    param_size dilation_factor = get<6>(GetParam());
    vpu::LayoutPreference layoutPreference = get<7>(GetParam());
    size_t out_w = (input_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;
    size_t out_h = (input_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;

    tensor_test_params output_dims = {1, out_channels, out_h, out_w};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    ie_fp16* bias = nullptr;

    std::map<std::string, std::string> layer_params = {
              {"kernel-x", std::to_string(kernel.x)}
            , {"kernel-y", std::to_string(kernel.y)}
            , {"stride-x", std::to_string(stride.x)}
            , {"stride-y", std::to_string(stride.y)}
            , {"pad-x", std::to_string(pad.x)}
            , {"pad-y", std::to_string(pad.y)}
            , {"output", std::to_string(out_channels)}
            , {"group", std::to_string(group)}
            , {"dilation-x", std::to_string(dilation_factor.x)}
            , {"dilation-y", std::to_string(dilation_factor.y)}
    };
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Convolution")
                                        .params(layer_params)
                                        .weights(num_weights),
                                        NetworkInitParams().layoutPreference(layoutPreference)
                                        .useHWOpt(true),
                                        weights_ptr));
    SetFirstInputToRange(-0.9f, 0.9f);

    ASSERT_TRUE(Infer());
}
