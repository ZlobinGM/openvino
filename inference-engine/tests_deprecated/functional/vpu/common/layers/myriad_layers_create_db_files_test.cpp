#include "myriad_layers_create_db_files_test_func.hpp"
#include "myriad_layers_create_db_files_test_param.hpp"

INSTANTIATE_TEST_CASE_P(sample_conv2120, myriadLayerConvolution_tests,
        ::testing::Combine(
            ::testing::ValuesIn(sampleIOChannels2120)
          , ::testing::ValuesIn(testDimsHW2120)
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)
           )
);
