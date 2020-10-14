#!/usr/bin/env python3
import os
import re
import sys
import random
import pandas as pd
import subprocess

from datetime import datetime

openvino_path = os.path.realpath(__file__)
for i in range(5): openvino_path = os.path.dirname(openvino_path)

def CheckDumpPerformance(openvino_path):
    file_to_check_path = os.path.join(openvino_path, 'inference-engine/tests_deprecated/functional/vpu/vpu_base/vpu_layers_tests.cpp')
    with open(file_to_check_path, 'r') as file_to_check:
        data = file_to_check.readlines()

    in_func_flag = 0
    string_found_flag = 0
    for line in data:
        if line.find('vpuLayersTests::Infer') != -1:
            in_func_flag = 1
            continue

        curr_line = line.find('dumpPerformance')
        if in_func_flag == 1 and curr_line != -1 and (line.find(r'//') > curr_line or line.find(r'//') == -1):
            string_found_flag = 1

        if in_func_flag == 1 and line == '}\n':
            break

    if string_found_flag == 0:
        print('Add in '+file_to_check_path+' in method Infer() a string "dumpPerformance();"')
        exit()
CheckDumpPerformance(openvino_path)

def CheckPerfcheckMain(openvino_path):
    file_to_check_path = os.path.join(openvino_path, 'inference-engine/tools/vpu/vpu_perfcheck/main.cpp')
    with open(file_to_check_path, 'r') as file_to_check:
        data = file_to_check.readlines()

    rxMinIter = r'#define MIN_ITER (?P<MIN_ITER>\d+)'
    rxProfile = r'const int profile = (?P<Profile>\d+);'
    for line in data:
        match = re.search(rxMinIter, line)
        if match is not None:
            min_iter = int(match.group('MIN_ITER'))
            if min_iter != 1:
                print('Set MIN_ITER in '+file_to_check_path+' to 1: #define MIN_ITER 1')
                exit()
            continue

        match = re.search(rxProfile, line)
        if match is not None:
            profile = int(match.group('Profile'))
            if profile != 1:
                print('Set profile in '+file_to_check_path+' to 1: const int profile = 1;')
                exit()
            continue
CheckPerfcheckMain(openvino_path)

if len(sys.argv)!=6:
    print('Incorrect args, should be: path_to_model'+os.linesep+
          'Example: ./some_path/model.xml ./some_path/weights.bin ./some_path/image_dir/ num_of_perfcheck_iterations num_of_contol_perfcheck_iterations'+os.linesep)
    exit()

path_to_model = sys.argv[1]
if not os.path.isfile(path_to_model):
    print('No such file ' + path_to_model)
    exit()

# rxWeightsBIN = r'\.bin'
# path_to_weights = sys.argv[2]
# if not os.path.isfile(path_to_weights) or re.search(rxWeightsBIN, path_to_weights) is None:
#     print('No such file ' + path_to_weights)
#     exit()

path_to_image_dir = sys.argv[3]
if not os.path.isdir(path_to_image_dir):
    print('No such dir ' + path_to_image_dir)
    exit()

num_iterations = sys.argv[4]
if not num_iterations.isdigit():
    print('Not a number')
    exit()
num_control_iterations = sys.argv[5]
if not num_iterations.isdigit():
    print('Not a number')
    exit()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# from openvino.inference_engine import IECore, IENetwork
# import ngraph as ng
# from ngraph.impl.op import Parameter
# from ngraph.impl import Function, Shape, Type

# def get_net(model: str, core: IECore):
#     model_xml = model
#     model_bin = os.path.splitext(model_xml)[0] + ".bin"
#     net = core.read_network(model=model_xml, weights=model_bin)
#     return net
# def load_mode(path_to_model: str):
#     core = IECore()
#     net = get_net(model=path_to_model, core=core)

#     func = ng.function_from_cnn(net)
#     ops = func.get_ordered_ops()    # & on value

#     i = 0
#     for op in ops:
#         if op.get_type_name() == 'Convolution':
#             in_shapes = [sh for sh in op.input(0).get_shape()]
#             out_shapes = [sh for sh in op.shape]
#             attrs = op._get_attributes()
#             pads_begin = attrs['pads_begin']
#             pads_end = attrs['pads_end']
#             if in_shapes[2:] == out_shapes[2:] and in_shapes[0] == out_shapes[0] == 1 and pads_begin == pads_end == [0,0]:
#                 print(i)
#                 i += 1
#                 print(op)
#                 print(op.get_rt_info())
#                 if(i == 59):
#                     op.get_rt_info()["TryToImproveReshape"] = "True"
#                     if op.get_rt_info()["TryToImproveReshape"] == "True":
#                         print(op.get_rt_info())
#                         print(op.get_rt_info()["TryToImproveReshape"])
#                         print(op.get_rt_info()["Variant::RuntimeAttribute::FusedNames"])

#     net_graph_file_xml = os.path.expanduser('~') + '/Downloads/net_graph.xml'
#     net_graph_file_bin = os.path.expanduser('~') + '/Downloads/net_graph.bin'
#     net.serialize(net_graph_file_xml, net_graph_file_bin)

#     net

# load_mode(path_to_model)


# def GetConv1x1FromXML(path_to_model):
#     rxFindNet = r'<net name=\"(?P<NetName>[\w\-]+)\"(?:(?: \w+=\"\w+\")+)?>'
#     rxFindConv = (
#     r'((?:\t+)?<layer id=\"\d+\" name=\"(?P<Name>[\w\/]+)\"(?:(?: \w+=\"\w+\")+)? type=\"Convolution\"(?:(?: \w+=\"\w+\")+)?>'+os.linesep+
#     r'(?:\t+)?<data(?:(?: \w+=\"[\w\,]+\")+)? strides="(?P<StrideX>\d+),(?P<StrideY>\d+)"(?:(?: \w+=\"[\w\,]+\")+)?\/>'+os.linesep+
#     r'(?:\t+)?<input>'+os.linesep+
#     r'(?:\t+)?<port id=\"0\"(?:(?: \w+=\"[\w\,]+\")+)?>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<InputTensorDimN>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<InputTensorDimC>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<InputTensorDimH>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<InputTensorDimW>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<\/port>'+os.linesep+
#     r'(?:\t+)?<port id=\"1\"(?:(?: \w+=\"[\w\,]+\")+)?>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<WeightsDimN>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<WeightsDimC>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<WeightsDimH>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<WeightsDimW>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<\/port>'+os.linesep+
#     r'(?:\t+)?<\/input>'+os.linesep+
#     r'(?:\t+)?<output>'+os.linesep+
#     r'(?:\t+)?<port id=\"2\"(?:(?: \w+=\"[\w\,]+\")+)?>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<OutputTensorDimN>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<OutputTensorDimC>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<OutputTensorDimH>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?<dim>(?P<OutputTensorDimW>\d+)<\/dim>'+os.linesep+
#     r'(?:\t+)?</port>'+os.linesep+
#     r'(?:\t+)?</output>'+os.linesep+
#     r'(?:\t+)?</layer>)')

#     with open(path_to_model) as dataFile:
#         data = dataFile.read()

#     netName = re.search(rxFindNet, data)
#     if (netName is None):
#         print('No net name')
#         netName = 'some_net'
#     else:
#         netName = netName.group('NetName')

#     matchAllConv = re.findall(rxFindConv, data)
#     if (matchAllConv is None):
#         print('No convolutions found')
#         exit()

#     listOfConv = []
#     for matchConv in matchAllConv:
#         match = re.search(rxFindConv, matchConv[0])

#         InN = int(match.group('InputTensorDimN'))
#         InC = int(match.group('InputTensorDimC'))
#         InH = int(match.group('InputTensorDimH'))
#         InW = int(match.group('InputTensorDimW'))

#         # WeightN = int(match.group('WeightsDimN'))
#         # WeightC = int(match.group('WeightsDimC'))
#         WeightH = int(match.group('WeightsDimH'))
#         WeightW = int(match.group('WeightsDimW'))

#         OutN = int(match.group('OutputTensorDimN'))
#         OutC = int(match.group('OutputTensorDimC'))
#         # OutH = int(match.group('OutputTensorDimH'))
#         # OutW = int(match.group('OutputTensorDimW'))

#         StrideX = int(match.group('StrideX'))
#         StrideY = int(match.group('StrideY'))

#         Name = match.group('Name')

#         if (InN != 1 or OutN != 1): continue
#         if (InH * InW == 1): continue
#         if (WeightH != 1 or WeightW != 1): continue
#         if (StrideX != 1 or StrideY != 1): continue

#         currentDict = {'Name' : Name, 'InC' : InC, 'OutC' : OutC, 'DimH' : InH, 'DimW' : InW}
#         listOfConv.append(currentDict) if currentDict not in listOfConv else listOfConv
#     return netName, listOfConv
# net_name, conv_list = GetConv1x1FromXML(path_to_model)

# i = 0
# for row in conv_list:
#     print(i)
#     i += 1
#     print(row)



# exit()
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

start = datetime.now()

print('\nWorking with model from: ' + path_to_model)

rxInputXML = r'\.xml'
rxInputONNX = r'\.onnx'
if re.search(rxInputXML, path_to_model) is None and re.search(rxInputONNX, path_to_model) is None:
    print('Path not to xml file or onnx file')
    exit()

def GetConv1x1(path_to_model, image_dir):
    make_perfcheck = subprocess.Popen('make myriad_perfcheck -j8',
                                        shell=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        cwd=os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape'))
    make_perfcheck.wait()

    rxName = r'(?:[\S]+)?\/(?P<Name>\S+)\.\w+'
    matchName = re.search(rxName, path_to_model)
    netName = 'some_model' if matchName is None else matchName.group('Name')
    perfcheck_path = os.path.join(openvino_path, 'bin/intel64/Release/myriad_perfcheck')
    out_path = os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/perfcheck('+netName+').txt')

    cmd = [perfcheck_path, path_to_model, image_dir, '1']
    with open(out_path,"w") as out:
        write_output = subprocess.Popen(cmd, stdout=out)
        write_output.wait()

    with open(out_path,"r") as out:
        data = out.readlines()

    rxFindConv = r'(?P<FIND_CONV>FIND_CONV(?:\w+)?_1X1) (?P<Name>[\w\/]+) InputC=(?P<InC>\d+) OutputC=(?P<OutC>\d+) DimH=(?P<DimH>\d+) DimW=(?P<DimW>\d+)'

    listOfConv = []
    for line in data:
        match = re.search(rxFindConv, line)

        if match is not None:
            InC = int(match.group('InC'))
            OutC = int(match.group('OutC'))
            DimH = int(match.group('DimH'))
            DimW = int(match.group('DimW'))
            Name = match.group('Name')
            currentDict = {'Name' : Name, 'InC' : InC, 'OutC' : OutC, 'DimH' : DimH, 'DimW' : DimW}
            listOfConv.append(currentDict) if currentDict not in listOfConv else listOfConv
    return netName, listOfConv

print('Get all conv 1x1 in net')
netName, listOfConv1x1 = GetConv1x1(path_to_model, path_to_image_dir)

df_AllConv1x1 = pd.DataFrame(listOfConv1x1)
print('\nAll convolutions 1x1 in net:')
print(df_AllConv1x1)
df_DiffConv1x1 = df_AllConv1x1.drop(columns=['Name']).drop_duplicates().reset_index(drop=True)
print('\nDifferent convolutions 1x1 in net:')
print(df_DiffConv1x1)

print('\nCreating tests')
def CreateTests(openvino_path, valueHW, dfConv):

    def dividers(val):
        d = []
        i = 1
        while (i * i <= val):
            if (val % i == 0):
                d.append(i)
                if (val // i != i):
                    d.append(val // i)
            i += 1
        return d

    test_param_path = os.path.join(openvino_path,
    'inference-engine/tests_deprecated/functional/vpu/common/layers/myriad_layers_improve_through_reshape_files_test_param.hpp')
    test_param = open(test_param_path, 'w')
    test_path = os.path.join(openvino_path,
    'inference-engine/tests_deprecated/functional/vpu/common/layers/myriad_layers_improve_through_reshape_files_test.cpp')
    test = open(test_path, 'w')

    test.write('#include "myriad_layers_improve_through_reshape_files_test_func.hpp"'+os.linesep)
    test.write('#include "myriad_layers_improve_through_reshape_files_test_param.hpp"'+os.linesep)
    test.write(os.linesep)

    test_param.write('static const std::initializer_list<std::vector<size_t>> testDimsHW'+str(valueHW)+' = {' + os.linesep)
    Diviedrs = dividers(valueHW)
    for H in Diviedrs:
        W = valueHW//H
        test_param.write('    {'+str(H)+', '+str(W)+'},'+os.linesep)
    test_param.write('};' + os.linesep)

    valuesIO = (dfConv[dfConv['DimH']*dfConv['DimW'] == valueHW])
    valuesIO = valuesIO[['InC', 'OutC']].drop_duplicates().values.tolist()
    test_param.write('static const std::initializer_list<std::vector<size_t>> testChannelsIOForHW'+str(valueHW)+' = {' + os.linesep)
    for [InC, OutC] in valuesIO:
        test_param.write('    {'+str(InC)+', '+str(OutC)+'},'+os.linesep)
    test_param.write('};' + os.linesep)

    test.write('INSTANTIATE_TEST_CASE_P(sample_conv'+str(valueHW)+', myriadLayerConvolution_tests,'+os.linesep)
    test.write('        ::testing::Combine('+os.linesep)
    test.write('            ::testing::ValuesIn(testChannelsIOForHW'+str(valueHW)+')'+os.linesep)
    test.write('          , ::testing::ValuesIn(testDimsHW'+str(valueHW)+')'+os.linesep)
    test.write('          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))'+os.linesep)
    test.write('          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))'+os.linesep)
    test.write('          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))'+os.linesep)
    test.write('          , ::testing::Values<group>(1)'+os.linesep)
    test.write('          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))'+os.linesep)
    test.write('          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)'+os.linesep)
    test.write('           )'+os.linesep)
    test.write(');' + os.linesep)

    test_param.close()
    test.close()
CreateTests(openvino_path, 0, df_DiffConv1x1)

def CreateTestFunc(openvino_path):
    test_func_path = os.path.join(openvino_path,
    'inference-engine/tests_deprecated/functional/vpu/common/layers/myriad_layers_improve_through_reshape_files_test_func.hpp')
    test_func = open(test_func_path, 'w')

    test_func.write('#include "myriad_layers_tests.hpp"'+os.linesep)
    test_func.write('#include "myriad_layers_reference_functions.hpp"'+os.linesep)
    test_func.write('#include "weights_for_convolution_test.h"'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('#include "conv_ref.hpp"'+os.linesep)
    test_func.write('using std::tuple;'+os.linesep)
    test_func.write('using std::get;'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('using namespace InferenceEngine;'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('PRETTY_PARAM(kernel, param_size);'+os.linesep)
    test_func.write('PRETTY_PARAM(stride, param_size);'+os.linesep)
    test_func.write('PRETTY_PARAM(pad, param_size);'+os.linesep)
    test_func.write('PRETTY_PARAM(out_channels, int);'+os.linesep)
    test_func.write('PRETTY_PARAM(group, int);'+os.linesep)
    test_func.write('PRETTY_PARAM(dilation_factor, param_size);'+os.linesep)
    test_func.write('PRETTY_PARAM(layoutPreference, vpu::LayoutPreference);'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('typedef myriadLayerTestBaseWithParam<tuple<std::vector<size_t>, std::vector<size_t>, kernel, stride, pad'+os.linesep)
    test_func.write('        , group,dilation_factor, layoutPreference >> myriadLayerConvolution_tests;'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('TEST_P(myriadLayerConvolution_tests, Convolution) {'+os.linesep)
    test_func.write('    std::vector<size_t> inputHW = get<1>(GetParam());'+os.linesep)
    test_func.write('    std::vector<size_t> IO = get<0>(GetParam());'+os.linesep)
    test_func.write('    size_t inputC = IO[0];'+os.linesep)
    test_func.write('    tensor_test_params input_dims = {1, inputC, inputHW[0], inputHW[1]};'+os.linesep)
    test_func.write('    param_size kernel = get<2>(GetParam());'+os.linesep)
    test_func.write('    param_size stride = get<3>(GetParam());'+os.linesep)
    test_func.write('    param_size pad = get<4>(GetParam());'+os.linesep)
    test_func.write('    size_t out_channels = IO[1];'+os.linesep)
    test_func.write('    size_t group = get<5>(GetParam());'+os.linesep)
    test_func.write('    param_size dilation_factor = get<6>(GetParam());'+os.linesep)
    test_func.write('    vpu::LayoutPreference layoutPreference = get<7>(GetParam());'+os.linesep)
    test_func.write('    size_t out_w = (input_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;'+os.linesep)
    test_func.write('    size_t out_h = (input_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('    tensor_test_params output_dims = {1, out_channels, out_h, out_w};'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('    SetInputTensor(input_dims);'+os.linesep)
    test_func.write('    SetOutputTensor(output_dims);'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;'+os.linesep)
    test_func.write('    size_t num_bias = output_dims.c;'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr ='+os.linesep)
    test_func.write('            InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));'+os.linesep)
    test_func.write('    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();'+os.linesep)
    test_func.write('    ie_fp16* bias = nullptr;'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('    std::map<std::string, std::string> layer_params = {'+os.linesep)
    test_func.write('              {"kernel-x", std::to_string(kernel.x)}'+os.linesep)
    test_func.write('            , {"kernel-y", std::to_string(kernel.y)}'+os.linesep)
    test_func.write('            , {"stride-x", std::to_string(stride.x)}'+os.linesep)
    test_func.write('            , {"stride-y", std::to_string(stride.y)}'+os.linesep)
    test_func.write('            , {"pad-x", std::to_string(pad.x)}'+os.linesep)
    test_func.write('            , {"pad-y", std::to_string(pad.y)}'+os.linesep)
    test_func.write('            , {"output", std::to_string(out_channels)}'+os.linesep)
    test_func.write('            , {"group", std::to_string(group)}'+os.linesep)
    test_func.write('            , {"dilation-x", std::to_string(dilation_factor.x)}'+os.linesep)
    test_func.write('            , {"dilation-y", std::to_string(dilation_factor.y)}'+os.linesep)
    test_func.write('    };'+os.linesep)
    test_func.write('    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Convolution")'+os.linesep)
    test_func.write('                                        .params(layer_params)'+os.linesep)
    test_func.write('                                        .weights(num_weights),'+os.linesep)
    test_func.write('                                        NetworkInitParams().layoutPreference(layoutPreference)'+os.linesep)
    test_func.write('                                        .useHWOpt(true),'+os.linesep)
    test_func.write('                                        weights_ptr));'+os.linesep)
    test_func.write('    SetFirstInputToRange(-0.9f, 0.9f);'+os.linesep)
    test_func.write(os.linesep)
    test_func.write('    ASSERT_TRUE(Infer());'+os.linesep)
    test_func.write('}' + os.linesep)
    test_func.close()
CreateTestFunc(openvino_path)

def DoTests(openvino_path, netName, valueHW):
    out_path = os.path.join(openvino_path,
    'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/auto_output('+netName+'_'+str(valueHW)+').txt')
    tests_path = os.path.join(openvino_path, 'bin/intel64/Release/MyriadFunctionalTests')
    test_name = 'sample_conv' + str(valueHW)
    filter_start = '--gtest_filter=*'+test_name+'*'

    rxEndTest = r'\[ RUN      \] ' + test_name + r'/myriadLayerConvolution_tests\.Convolution/(?P<TestNum>\d+)'
    rxEndTest = r'\[ RUN      \] ' + test_name + r'/myriadLayerConvolution_tests\.Convolution/(?P<TestNum>\d+)'
    rxTotalTests= r'\[==========\] Running (?P<TotalTests>\d+) tests from \d+ test case'
    rxPass = r'\[  PASSED  \]'

    def FindLastTest():
        LastNum = 0

        if not os.path.isfile(out_path):
            return LastNum

        flag_passed = False
        for line in reversed(open(out_path).readlines()):
            LastNumString = re.search(rxEndTest, line)
            PassAll = re.search(rxPass, line)
            if PassAll is not None:
                flag_passed = True
            if LastNumString is not None:
                LastNum = int(LastNumString[1])
                if (flag_passed): LastNum += 1
                return LastNum
        return LastNum

    def FindTotalTests():
        TotalTests = 10**6

        if not os.path.isfile(out_path):
            return TotalTests

        for line in open(out_path).readlines():
            TotalTestsString = re.search(rxTotalTests, line)
            if TotalTestsString is not None:
                TotalTests = int(TotalTestsString[1])
                return TotalTests
        return TotalTests

    def CreateFilter(lastTest):
        cr_filter = '-' if lastTest!=0 else ''

        strTest = str(lastTest)

        for i in range(len(strTest)):
            cr_filter = cr_filter if i == 0 else cr_filter+'*/'+'?'*(i)+':'

            filter_start = '' if i == 0 else strTest[:i]
            value = int(strTest[i])

            for lessNum in range(value):
                if(len(strTest) == 1 and lessNum == 0): cr_filter = cr_filter+'*/0:'
                if(i == 0 and lessNum == 0): continue
                cr_filter = cr_filter+'*/'+filter_start+str(lessNum)+'?'*(len(strTest)-len(filter_start)-1)+':'

        return cr_filter

    numOfTests = 0
    while 1:
        lastTest = FindLastTest()
        lastTest = lastTest if lastTest==0 else lastTest+1
        print('Last run test: '+str(lastTest))
        numOfTests = FindTotalTests()
        if lastTest >= (numOfTests-1):
            numOfTests = lastTest
            break

        new_filter = CreateFilter(lastTest)
        cmd = [tests_path, filter_start+new_filter]
        print('gtests filter: '+str(new_filter))
        with open(out_path,"a") as out:
            write_output = subprocess.Popen(cmd, stdout=out)
            write_output.wait()

    print('End with total tests: '+str(numOfTests))

def ParseOutputToCSV(netName, valueHW):
    rxInDims = r'\[ VALUE    \]\s+\(\{ (?P<InC>\d+)\, (?P<OutC>\d+) \}\, \{ (?P<DimH>\d+)\, (?P<DimW>\d+) \}\, kernel'
    rxTimeConv = r'type is (?P<TimeConv>\d+\.\d+) ms'
    rxBegin = r'\[ RUN      \]'
    rxEnd = r'\[ VALUE    \]'
    rxFail = r'\[  FAILED  \]'
    rxPass = r'\[  PASSED  \]'

    path_to_txt = os.path.join(openvino_path,
    'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/auto_output('+netName+'_'+str(valueHW)+').txt')

    with open(path_to_txt) as dataFile:
        data=dataFile.readlines()

    InC = []
    DimH = []
    DimW = []
    OutC = []
    TimeConv = []
    sumTime = 0
    nulls = 0
    for line in data:
        start = re.search(rxBegin, line)
        if start is not None:
            sumTime = 0

        Dims = re.search(rxInDims, line)
        if Dims is not None:
            InC.append(int(Dims.group('InC')))
            OutC.append(int(Dims.group('OutC')))
            DimH.append(int(Dims.group('DimH')))
            DimW.append(int(Dims.group('DimW')))

        Time = re.search(rxTimeConv, line)
        if Time is not None:
            sumTime+=float(Time[1])*(10**6)

        end = re.search(rxEnd, line)
        if end is not None:
            TimeConv.append(sumTime/(10**6))
            if sumTime == 0:
                nulls +=1

        fail = re.search(rxFail, line)
        if fail is not None:
            del InC[-1]
            del DimH[-1]
            del DimW[-1]
            del OutC[-1]
            del TimeConv[-1]

        passed = re.search(rxPass, line)
        if passed is not None:
            break

    path_to_csv = os.path.join(openvino_path,
    'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/auto_output('+netName+'_'+str(valueHW)+').csv')

    df = pd.DataFrame({'InC' : InC, 'OutC' : OutC, 'DimH' : DimH, 'DimW' : DimW,
                    'TimeConv' : TimeConv})
    df = df[df['TimeConv']!=0]
    df.to_csv(path_to_csv, index=False, encoding='utf-8')

    return df

if not os.path.isdir(os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape')):
    print('\nCreating cmake-build-for-improve_through_reshape dir')
    os.mkdir(os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape'))

print('\nRunning cmake')
cmake_FuncTests = subprocess.Popen('cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON -DENABLE_GNA=OFF -DENABLE_CLDNN=OFF -DENABLE_MKL_DNN=OFF',
                                    shell=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    cwd=os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape'))
cmake_FuncTests.wait()

if not os.path.isdir(os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape', 'temp-files-for-improve_through_reshape')):
    print('\nCreating in cmake-build-for-improve_through_reshape dir for temp txt and csv files')
    os.mkdir(os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape', 'temp-files-for-improve_through_reshape'))

df_AllTimeConv1x1 = pd.DataFrame()
valuesHW = (df_DiffConv1x1['DimH']*df_DiffConv1x1['DimW']).drop_duplicates().to_list()
for valueHW in valuesHW:
    print('\nCreating test for HW = '+str(valueHW))
    CreateTests(openvino_path, valueHW, df_DiffConv1x1)

    print('Running make MyriadFunctionalTests')
    make_FuncTests = subprocess.Popen('make MyriadFunctionalTests -j8',
                                        shell=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        cwd=os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape'))
    make_FuncTests.wait()

    print('Running tests')
    DoTests(openvino_path, netName, valueHW)
    print('Parsing to csv')
    df_AllTimeConv1x1 = pd.concat([df_AllTimeConv1x1, ParseOutputToCSV(netName, valueHW)])

df_AllTimeConv1x1['HW'] = df_AllTimeConv1x1['DimH'] * df_AllTimeConv1x1['DimW']
merge_on = list(df_DiffConv1x1.columns)
df_AllConv1x1 = pd.merge(df_AllConv1x1, df_AllTimeConv1x1, how='left',
                        left_on=merge_on, right_on=merge_on)
df_DiffConv1x1 = pd.merge(df_DiffConv1x1, df_AllTimeConv1x1, how='left',
                        left_on=merge_on, right_on=merge_on)

print('\nCurrent convolutions 1x1 times in net with one convolution:')
print(df_DiffConv1x1)

def LongOptimize(df_CurrentConv1x1, df_TimesConv1x1, openvino_path, path_to_xml, image_dir, num_iters, num_control_iters):
    def ResetImplementedRow():
        path_to_pass = os.path.join(openvino_path, 'inference-engine/src/vpu/graph_transformer/include/vpu' +
                        '/middleend/hw/conv_tiling/reshape_conv_func.hpp')
        with open(path_to_pass, 'r') as passFile:
            data = passFile.readlines()

        df_ImplementedRows = pd.DataFrame()
        rxRowInPass = r'(?P<Comment>\/\/(?:\s+)?)?if \(\(Name == \"(?P<Name>[\w\/]+)\"\) && \(DimH \* DimW == (?P<HW>[\d\.]+)\) && \(InC == (?P<InC>[\d\.]+)\) && \(OutC == (?P<OutC>[\d\.]+)\)\) return (?P<DimH>[\d\.]+);'
        for line in data:
            match = re.search(rxRowInPass, line)
            if match is None: continue
            if match.group('Comment') is not None: continue
            Name = match.group('Name')
            InC = int(float(match.group('InC')))
            OutC = int(float(match.group('OutC')))
            HW = int(float(match.group('HW')))
            DimH = int(float(match.group('DimH')))
            DimW = HW // DimH
            match_dict = {'Name' : Name, 'InC' : InC, 'OutC' : OutC, 'DimH' : DimH, 'DimW' : DimW, 'HW' : HW}
            df_ImplementedRows = df_ImplementedRows.append(pd.Series(match_dict), ignore_index=True)

        return df_ImplementedRows
    def ImplementRow(new_row):
        df_ImplementedRows = ResetImplementedRow()
        if len(df_ImplementedRows) > 0:
            row_same_if = df_ImplementedRows[df_ImplementedRows['Name'] == new_row['Name']]
            row_same = row_same_if[row_same_if['DimH'] == new_row['DimH']]

            if len(row_same_if) > 1 and len(row_same) != 0:
                for _index, row in row_same_if.iterrows():
                    DeleteRow(row)
            if len(row_same) != 0: return
            if len(row_same_if) != 0:
                for _index, row in row_same_if.iterrows():
                    DeleteRow(row)

        path_to_pass = os.path.join(openvino_path, 'inference-engine/src/vpu/graph_transformer/include/vpu' +
                        '/middleend/hw/conv_tiling/reshape_conv_func.hpp')
        with open(path_to_pass, 'r') as passFile:
            data = passFile.readlines()


        rxEndOfPass = r'return 0;'
        with open(path_to_pass, 'w') as passFile:
            for line in data:
                match = re.search(rxEndOfPass, line)
                if match is None:
                    passFile.write(line)
                    continue
                passFile.write('    if ((Name == "'+(new_row['Name'])+'") && (DimH * DimW == '+str(round(new_row['HW']))+') && (InC == '+
                                str(round(new_row['InC']))+') && (OutC == '+str(round(new_row['OutC']))+')) return '+
                                str(round(new_row['DimH']))+';'+os.linesep)
                passFile.write(line)
    def DeleteRow(row_to_delete):
        df_ImplementedRows = ResetImplementedRow()
        if len(df_ImplementedRows) == 0: return

        row_to_delete = df_ImplementedRows[df_ImplementedRows['Name'] == row_to_delete['Name']]
        if len(row_to_delete) == 0: return

        row_to_delete = row_to_delete.iloc[0]

        path_to_pass = os.path.join(openvino_path, 'inference-engine/src/vpu/graph_transformer/include/vpu' +
                        '/middleend/hw/conv_tiling/reshape_conv_func.hpp')
        with open(path_to_pass, 'r') as passFile:
            data = passFile.readlines()

        rxRowInPass = r'(?P<Comment>\/\/(?:\s+)?)?if \(\(Name == \"(?P<Name>[\w\/]+)\"\) && \(DimH \* DimW == (?P<HW>[\d\.]+)\) && \(InC == (?P<InC>[\d\.]+)\) && \(OutC == (?P<OutC>[\d\.]+)\)\) return (?P<DimH>[\d\.]+);'
        with open(path_to_pass, 'w') as passFile:
            for line in data:
                match = re.search(rxRowInPass, line)
                if match is None:
                    passFile.write(line)
                    continue
                if match.group('Comment') is not None:
                    passFile.write(line)
                    continue
                Name = match.group('Name')
                InC = int(float(match.group('InC')))
                OutC = int(float(match.group('OutC')))
                HW = int(float(match.group('HW')))
                if (Name == row_to_delete['Name'] and InC == row_to_delete['InC'] and OutC == row_to_delete['OutC'] and HW == row_to_delete['HW']):
                    continue
                passFile.write(line)

    def DoPerfcheck(name, path_to_xml, image_dir, num_iters):
        make_perfcheck = subprocess.Popen('make myriad_perfcheck -j8',
                                            shell=True,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                            cwd=os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape'))
        make_perfcheck.wait()

        perfcheck_path = os.path.join(openvino_path, 'bin/intel64/Release/myriad_perfcheck')
        out_path = os.path.join(openvino_path,
        'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/perfcheck('+netName+'_'+name+').txt')

        if isinstance(num_iters, int): num_iters = str(num_iters)
        cmd = [perfcheck_path, path_to_xml, image_dir, num_iters]
        with open(out_path,"w") as out:
            write_output = subprocess.Popen(cmd, stdout=out)
            try:
                write_output.wait(300) # wait 300 seconds
            except subprocess.TimeoutExpired:
                write_output.kill()
                return 0, 0, None, None, None

        with open(out_path,"r") as out:
            data = out.readlines()

        Time = 0
        FPS = 0
        rxInfTime = r'Total inference time:\s+(?P<Time>[\d\.]+)'
        rxAvgFPS = r'Average fps on \d+ iterations: (?P<FPS>[\d\.]+) fps'
        for line in data:
            matchTime = re.search(rxInfTime, line)
            matchFPS = re.search(rxAvgFPS, line)
            if matchTime is not None:
                Time = float(matchTime.group('Time'))
            if matchFPS is not None:
                FPS = float(matchFPS.group('FPS'))

        rxFindConv = r'(?P<FIND_CONV>FIND_CONV(?:\w+)?_1X1) (?P<Name>[\w\/]+) InputC=(?P<InC>\d+) OutputC=(?P<OutC>\d+) DimH=(?P<DimH>\d+) DimW=(?P<DimW>\d+)'
        rxStageTime = r'(?P<Index>\d+)\s+(?P<Name>(?:\S+ )+)\s+(?P<Type>(?:\S+ )+)\s+(?P<Time>\d+\.\d+)'

        fcInC = []
        fcOutC = []
        fcDimH = []
        fcDimW = []
        fcName = []
        fcIsTarget = []

        stName = []
        stType = []
        stTime = []

        for line in data:
            match_fc = re.search(rxFindConv, line)
            if match_fc is not None:
                fcInC.append(int(match_fc.group('InC')))
                fcOutC.append(int(match_fc.group('OutC')))
                fcDimH.append(int(match_fc.group('DimH')))
                fcDimW.append(int(match_fc.group('DimW')))
                fcName.append(match_fc.group('Name'))
                fcIsTarget.append(match_fc.group('FIND_CONV') == 'FIND_CONV_FOR_RESHAPE_1X1')

            match_st = re.search(rxStageTime, line)
            if match_st is not None:
                stName.append(match_st.group('Name'))
                stType.append(match_st.group('Type'))
                stTime.append(float(match_st.group('Time')))

        df_FindConv = pd.DataFrame({'Name' : fcName, 'InC' : fcInC, 'OutC' : fcOutC, 'DimH' : fcDimH,
                        'DimW' : fcDimW, 'IsTarget' : fcIsTarget})
        df_StageTime = pd.DataFrame({'Name' : stName, 'Type' : stType, 'Time' : stTime})

        df_TypesTime = df_StageTime.drop(columns=['Name']).groupby(['Type'])['Time'].agg(TimeSum='sum', TimeCount='count')

        rxGroup = r'(?:injected\[)?(?P<Name>(?:\S+?))(?:\@\S+)? (?:\+)?'
        Group = []
        for _index, row in df_StageTime.iterrows():
            match_group = re.match(rxGroup, row['Name'])
            if match_group is not None:
                Group.append(match_group.group('Name'))
            else: Group.append('ERROR')
        df_StageTime['Group'] = Group
        df_OrigStageTime = df_StageTime.drop(columns=['Name', 'Type']).groupby(['Group']).sum()
        df_Conv1x1Time = pd.merge(df_FindConv, df_OrigStageTime, how='left', left_on=['Name'], right_on=['Group'])

        df_TargetGroups = pd.merge(df_StageTime, df_FindConv[['Name', 'IsTarget']], how='left', left_on=['Group'], right_on=['Name'], suffixes=('', '_y'))
        df_TargetGroups = df_TargetGroups.drop(columns=['Name_y'])

        rxSplitHWC = r'(?:(?:\@sow=\d+\/(?P<sow>\d+))|(?:\@soh=\d+\/(?P<soh>\d+))|(?:\@soc=\d+\/(?P<soc>\d+)))+'
        soh = []
        sow = []
        soc = []
        for _index, row in df_StageTime.iterrows():
            match_split = re.search(rxSplitHWC, row['Name'])
            if match_split is not None:
                soh.append(int(match_split.group('soh')) if match_split.group('soh') is not None else 1)
                sow.append(int(match_split.group('sow')) if match_split.group('sow') is not None else 1)
                soc.append(int(match_split.group('soc')) if match_split.group('soc') is not None else 1)
            else:
                soh.append(1)
                sow.append(1)
                soc.append(1)
        df_StageTime['soh'] = soh
        df_StageTime['sow'] = sow
        df_StageTime['soc'] = soc
        df_StageSplit = df_StageTime.drop(columns=['Name', 'Type', 'Time']).groupby(['Group']).max()
        df_Conv1x1Time = pd.merge(df_Conv1x1Time, df_StageSplit, how='left', left_on=['Name'], right_on=['Group'])

        df_ConvGroups = df_StageTime[['Group', 'Name']]
        return Time, FPS, df_TypesTime, df_Conv1x1Time, df_ConvGroups, df_TargetGroups

    def CompareConv(oldNetTime, newNetTime, oldTimes, newTimes, name, thresholdNetTime, thresholdTimes):
        oldTime = oldTimes[oldTimes['Name'] == name].iloc[0]['Time']
        newTime = newTimes[newTimes['Name'] == name].iloc[0]['Time']
        relTime = oldTime / newTime
        if relTime > (1 + thresholdTimes) and newNetTime < oldNetTime * (1 + thresholdNetTime): return True, relTime
        return False, relTime

    temp_df = ResetImplementedRow()
    for _index, row in temp_df.iterrows():
        DeleteRow(row)

    print('\nGetting current performance')
    currentTime, currentFPS, oldTypesTimes, oldStagesTimes, oldConvGroups, oldTargetGroups = DoPerfcheck('', path_to_xml, image_dir, num_control_iters)
    while currentTime == 0:
        currentTime, currentFPS, oldTypesTimes, oldStagesTimes, oldConvGroups, oldTargetGroups = DoPerfcheck('', path_to_xml, image_dir, num_control_iters)
    oldTime, oldFPS = currentTime, currentFPS
    print('Current Time: '+str(currentTime))
    print('Current FPS: '+str(currentFPS))

    for _index, row in df_CurrentConv1x1.iterrows():
        df_BetterTimes = df_TimesConv1x1[((df_TimesConv1x1['HW'] == row['HW']) &
                                        (df_TimesConv1x1['InC'] == row['InC']) &
                                        (df_TimesConv1x1['OutC'] == row['OutC']) &
                                        (df_TimesConv1x1['TimeConv'] < row['TimeConv']*0.95))]
        if len(df_BetterTimes) == 0:
            print('\nNo better convolutions 1x1 to try for Name = '+row['Name']+' , HW = '+str(row['HW'])+' , InC = '+str(row['InC'])+' , OutC  = '+str(row['OutC']))
            continue
        df_BetterTimes = df_BetterTimes.sort_values(by=['TimeConv'])
        df_BetterTimes['Name'] = row['Name']

        print('\nBetter convolutions 1x1 to try for Name = '+row['Name']+' , HW = '+str(row['HW'])+' , InC = '+str(row['InC'])+' , OutC  = '+str(row['OutC'])+' :')
        print(df_BetterTimes.drop(columns=['Name']))

        bestRow = row
        bestTime = currentTime
        bestFPS = currentFPS
        bestRelTime = 1
        bestStagesTimes = oldStagesTimes
        name = str(round(row['HW']))
        for _indexTry, rowTry in df_BetterTimes.iterrows():
            print('Try DimH='+str(rowTry['DimH'])+' DimW='+str(rowTry['DimW'])+
                ' for InC='+str(rowTry['InC'])+' OutC='+str(rowTry['OutC']))
            ImplementRow(rowTry)
            tempTime, tempFPS, tempTypesTimes, tempStagesTimes, tempConvGroups, tempTargetGroups = DoPerfcheck(name+'-'+str(round(rowTry['DimH']))+':'+str(round(rowTry['DimW'])),
                                                                                            path_to_xml, image_dir, num_iters)

            tempIsBetter, tempRelTime = CompareConv(bestTime, tempTime, bestStagesTimes, tempStagesTimes, row['Name'], 0.01, 0.05)

            if tempIsBetter and tempRelTime > bestRelTime:
                bestTime = tempTime
                bestFPS = tempFPS
                bestRow = rowTry
                bestRelTime = tempRelTime
                bestStagesTimes = tempStagesTimes
            DeleteRow(rowTry)
        if bestRelTime > 1:
            ImplementRow(bestRow)
            currentTime = bestTime
            currentFPS = bestFPS
            print('Better reshape: DimH='+str(bestRow['DimH'])+' DimW='+str(bestRow['DimW']))
            print('Better Time: '+str(currentTime))
            print('Better FPS: '+str(currentFPS))
        else: print('No better reshape')

    print('\nOld Time: '+str(oldTime))
    print('Old FPS: '+str(oldFPS))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        print('\nOld convolution stages times:')
        print(oldStagesTimes.drop(columns=['IsTarget']))
    newTime, newFPS, newTypesTimes, newStagesTimes, newConvGroups, newTargetGroups = DoPerfcheck(name, path_to_xml, image_dir, num_control_iters)
    while newTime == 0:
        newTime, newFPS, newTypesTimes, newStagesTimes, newConvGroups, newTargetGroups = DoPerfcheck(name, path_to_xml, image_dir, num_control_iters)
    print('\nNew Time: '+str(newTime))
    print('New FPS: '+str(newFPS))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        print('\nNew convolution stages times:')
        print(newStagesTimes.drop(columns=['IsTarget']))

    def TableCompareTypesTimes(oldTypesTimes, newTypesTimes):
        oldTypesTimes = oldTypesTimes.rename(columns={'TimeSum': 'OldTime', 'TimeCount': 'OldTimeCount'})
        newTypesTimes = newTypesTimes.rename(columns={'TimeSum': 'NewTime', 'TimeCount': 'NewTimeCount'})
        compareTypesTimes = pd.merge(oldTypesTimes, newTypesTimes, on=['Type']).fillna(0)
        compareTypesTimes['RelTimeCompare'] = (compareTypesTimes['OldTime'] / compareTypesTimes['NewTime']).round(2)
        compareTypesTimes['AbsTimeCompare'] = compareTypesTimes['OldTime'] - compareTypesTimes['NewTime']
        return compareTypesTimes.sort_values(by=['RelTimeCompare'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        print('\nCompare stages times by types:')
        print(TableCompareTypesTimes(oldTypesTimes, newTypesTimes))

    def TableCompareStagesTimes(oldStagesTimes, newStagesTimes):
        if (oldStagesTimes['Name'] == newStagesTimes['Name']).all():
            oldStagesTimes['IsTarget'] = newStagesTimes['IsTarget']
        newStagesTimes = newStagesTimes[newStagesTimes['IsTarget'] == True].drop(columns=['IsTarget'])
        newStagesTimes = newStagesTimes.rename(columns={'Time': 'NewTime', 'DimH' : 'NewDimH', 'DimW' : 'NewDimW', 'soh' : 'NewSOH',
                                                        'sow' : 'NewSOW', 'soc' : 'NewSOC'})
        NewDimH = []
        NewDimW = []
        temp_df = ResetImplementedRow()
        for _index, row in newStagesTimes.iterrows():
            df = temp_df[(temp_df['InC'] == row['InC']) & (temp_df['OutC'] == row['OutC'])]
            if len(df) != 0:
                temp_row = df.iloc[0]
                NewDimH.append(temp_row['DimH'])
                NewDimW.append(temp_row['DimW'])
            else:
                NewDimH.append(row['DimH'])
                NewDimW.append(row['DimW'])
        newStagesTimes['NewDimH'] = NewDimH
        newStagesTimes['NewDimW'] = NewDimW
        newStagesTimes = newStagesTimes.astype({'NewDimH' : int, 'NewDimW' : int})

        oldStagesTimes = oldStagesTimes[oldStagesTimes['IsTarget'] == True].drop(columns=['IsTarget'])
        oldStagesTimes = oldStagesTimes.rename(columns={'Time': 'OldTime', 'DimH' : 'OldDimH', 'DimW' : 'OldDimW', 'soh' : 'OldSOH',
                                                        'sow' : 'OldSOW', 'soc' : 'OldSOC'})

        compareTargetStages = pd.merge(oldStagesTimes, newStagesTimes, how='right', on=['Name', 'InC', 'OutC'])
        compareTargetStages.astype({'NewDimH': int, 'NewDimW': int})
        compareTargetStages['RelTimeCompare'] = (compareTargetStages['OldTime'] / compareTargetStages['NewTime']).round(2)
        compareTargetStages['AbsTimeCompare'] = compareTargetStages['OldTime'] - compareTargetStages['NewTime']
        return compareTargetStages.sort_values(by=['RelTimeCompare'])
    compareTargetStages = TableCompareStagesTimes(oldStagesTimes, newStagesTimes)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        Table = compareTargetStages[['Name', 'InC', 'OutC', 'OldDimH', 'OldDimW', 'OldSOH', 'OldSOW',
                                    'OldSOC', 'NewDimH', 'NewDimW', 'NewSOH', 'NewSOW', 'NewSOC']].drop_duplicates()
        if len(Table) > 0:
            print('\nCompare tilings:')
            print(Table)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        Table = compareTargetStages[['Name', 'InC', 'OutC', 'OldDimH', 'OldDimW', 'NewDimH', 'NewDimW',
                                            'OldTime', 'NewTime', 'RelTimeCompare', 'AbsTimeCompare']]
        if len(Table) > 0:
            print('\nCompare reshaped stages times:')
            print(Table)

    print('\nNew reshape dims:')
    print(ResetImplementedRow())

    def TableCompareTargetGroups(oldTargetGroups, newTargetGroups):
        if oldTargetGroups is None or len(newTargetGroups) is None: return []
        if len(oldTargetGroups) == 0 or len(newTargetGroups) == 0: return []
        targetGroups = newTargetGroups[newTargetGroups['IsTarget'] == True]['Group'].drop_duplicates().tolist()

        oldTargetGroups = oldTargetGroups.drop(columns=['IsTarget'])
        newTargetGroups = newTargetGroups.drop(columns=['IsTarget'])

        resultTable = pd.DataFrame()

        for curGroup in targetGroups:
            oldGroup = oldTargetGroups[oldTargetGroups['Group'] == curGroup].copy()
            oldGroup['Ver'] = 'Old'
            newGroup = newTargetGroups[newTargetGroups['Group'] == curGroup].copy()
            newGroup['Ver'] = 'New'
            resultTable = pd.concat([resultTable, oldGroup, newGroup])

        resultTable = resultTable.sort_values(by=['Group', 'Ver', 'Name'])
        resultTable = resultTable[['Group', 'Ver', 'Name', 'Type', 'Time']]
        return resultTable
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        Table = TableCompareTargetGroups(oldTargetGroups, newTargetGroups)
        if len(Table) > 0:
            print('\nCompare target groups:')
            print(Table)

LongOptimize(df_AllConv1x1, df_AllTimeConv1x1, openvino_path, path_to_model, path_to_image_dir, num_iterations, num_control_iterations)

end = datetime.now()

print(start)
print(end)
print(end-start)

print('Done')