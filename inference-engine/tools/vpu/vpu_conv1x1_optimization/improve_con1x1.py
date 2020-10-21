#!/usr/bin/env python3
import os
import re
import sys
import random
import pandas as pd
import subprocess
import xml.etree.ElementTree as ET

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

if len(sys.argv)!=5:
    print('Incorrect args, should be: path_to_model'+os.linesep+
          'Example: ./some_path/model.xml ./some_path/image_dir/ num_of_perfcheck_iterations num_of_contol_perfcheck_iterations'+os.linesep)
    exit()

path_to_model = sys.argv[1]
if not os.path.isfile(path_to_model):
    print('No such file ' + path_to_model)
    exit()

path_to_image_dir = sys.argv[2]
if not os.path.isdir(path_to_image_dir):
    print('No such dir ' + path_to_image_dir)
    exit()

num_iterations = sys.argv[3]
if not num_iterations.isdigit():
    print('Not a number')
    exit()
num_control_iterations = sys.argv[4]
if not num_iterations.isdigit():
    print('Not a number')
    exit()

start = datetime.now()

print('\nWorking with model from: ' + path_to_model)

not_ngraph = True
if not_ngraph:
    rxInputXML = r'\.xml'
    rxInputONNX = r'\.onnx'
    if re.search(rxInputXML, path_to_model) is None and re.search(rxInputONNX, path_to_model) is None:
        print('Path not to xml file or onnx file')
        exit()

    def parse_xml(path_to_model: str):
        tree = ET.parse(path_to_model)
        root = tree.getroot()
        layers = root.find('layers')

        netName = root.get('name')
        listOfConv = []

        for layer in layers.iter('layer'):
            if layer.get('type') == 'Convolution':
                data = layer.find('data')

                strides = data.get('strides').split(',')
                strides = map(int, strides)
                strides = list(strides)
                if strides != [1,1]: continue

                input = layer.find('input')
                if input is None: continue

                flag_skip = False
                inC = 0
                outC = 0
                dimH = 0
                dimW = 0
                for port in input:
                    if port.get('id') == '0':
                        dims = list(port)

                        inC = int(dims[1].text)
                        dimH = int(dims[2].text)
                        dimW = int(dims[3].text)

                        if dims[0].text != '1' or dims[2].text == dims[3].text == '1':
                            flag_skip = True
                            break
                    elif port.get('id') == '1':
                        dims = list(port)

                        outC = int(dims[0].text)

                        if dims[2].text != '1' or dims[3].text != '1':
                            flag_skip = True
                            break
                if flag_skip: continue
                name = layer.get('name')

                currentDict = {'Name' : name, 'InC' : inC, 'OutC' : outC, 'DimH' : dimH, 'DimW' : dimW}
                listOfConv.append(currentDict) if currentDict not in listOfConv else listOfConv
        return netName, listOfConv

    print('Parse .xml')
    netName, listOfConv1x1 = parse_xml(path_to_model)

    print('\nAll convolutions 1x1 in net:')
    df_AllConv1x1 = pd.DataFrame(listOfConv1x1)
    print(df_AllConv1x1)

    print('\nDifferent convolutions 1x1 in net:')
    df_DiffConv1x1 = df_AllConv1x1.drop(columns=['Name']).drop_duplicates().reset_index(drop=True)
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

    def create_model_copy(path_to_xml: str):
        path_to_xml_copy = os.path.splitext(path_to_xml)[0] + "_improve_conv1x1.xml"
        path_to_bin = os.path.splitext(path_to_xml)[0] + ".bin"
        path_to_bin_copy = os.path.splitext(path_to_xml_copy)[0] + ".bin"

        import shutil
        shutil.copy2(path_to_xml, path_to_xml_copy)
        shutil.copy2(path_to_bin, path_to_bin_copy)
        return path_to_xml_copy
    path_to_model = create_model_copy(path_to_model)

    def LongOptimize(df_CurrentConv1x1, df_TimesConv1x1, openvino_path, path_to_xml, image_dir, num_iters, num_control_iters):
        def get_implemented_conv(tree):
            root = tree.getroot()
            layers = root.find('layers')

            listOfConv = []

            for layer in layers.iter('layer'):
                data = layer.find('data')
                if data is not None and data.get('ConvReshape') is not None:
                    input = layer.find('input')

                    inC = 0
                    outC = 0
                    dimH = 0
                    dimW = 0
                    for port in input:
                        if port.get('id') == '0':
                            dims = list(port)

                            inC = int(dims[1].text)
                            dimH = int(dims[2].text)
                            dimW = int(dims[3].text)
                        elif port.get('id') == '1':
                            dims = list(port)

                            outC = int(dims[0].text)
                    hw = dimH * dimW

                    try:
                        dimW = int(data.get('ConvReshape'))
                    except ValueError:
                        dimW = 1
                    dimH = hw / dimW
                    name = layer.get('name')

                    currentDict = {'Name' : name, 'InC' : inC, 'OutC' : outC, 'DimH' : dimH, 'DimW' : dimW, 'HW' : hw}
                    listOfConv.append(currentDict) if currentDict not in listOfConv else listOfConv
            return pd.DataFrame(listOfConv)
        def implement_conv(tree, new_conv):
            implemented_convs = get_implemented_conv(tree)
            if len(implemented_convs) > 0:
                same_conv = implemented_convs[implemented_convs['Name'] == new_conv['Name']]
                if len(same_conv) != 0:
                    for _index, row in same_conv.iterrows():
                        revert_conv(tree, row)

            root = tree.getroot()
            layers = root.find('layers')

            for layer in layers.iter('layer'):
                if layer.get('name') == new_conv['Name']:
                    data = layer.find('data')
                    data.set('ConvReshape', str(new_conv['DimW']))
        def revert_conv(tree, del_conv):
            implemented_convs = get_implemented_conv(tree)
            if len(implemented_convs) == 0: return

            conv_to_delete = implemented_convs[implemented_convs['Name'] == del_conv['Name']]
            if len(conv_to_delete) == 0: return

            root = tree.getroot()
            layers = root.find('layers')

            for layer in layers.iter('layer'):
                if layer.get('name') == del_conv['Name']:
                    data = layer.find('data')
                    data.attrib.pop("ConvReshape", None)

        def perfcheck(netName, name, tree, path_to_xml, image_dir, num_iters):
            tree.write(path_to_xml)
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

            rxStageTime = r'(?P<Index>\d+)\s+(?P<Name>(?:\S+ )+)\s+(?P<Type>(?:\S+ )+)\s+(?P<Time>\d+\.\d+)'

            stName = []
            stType = []
            stTime = []

            for line in data:
                match_st = re.search(rxStageTime, line)
                if match_st is not None:
                    stName.append(match_st.group('Name'))
                    stType.append(match_st.group('Type'))
                    stTime.append(float(match_st.group('Time')))

            df_StageTime = pd.DataFrame({'Name' : stName, 'Type' : stType, 'Time' : stTime})

            df_TypesTime = df_StageTime.drop(columns=['Name']).groupby(['Type'])['Time'].agg(TimeSum='sum', TimeCount='count')

            rxGroup = r'(?P<Name>(?:\S+?))(?:\@\S+) (?:\+)?'
            Group = []
            for _index, row in df_StageTime.iterrows():
                match_group = re.match(rxGroup, row['Name'])
                if match_group is not None:
                    Group.append(match_group.group('Name'))
                else: Group.append(row['Name'])
            df_StageTime['Group'] = Group

            return Time, FPS, df_TypesTime, df_StageTime

        make_perfcheck = subprocess.Popen('make myriad_perfcheck -j8',
                                            shell=True,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                            cwd=os.path.join(openvino_path, 'cmake-build-for-improve_through_reshape'))
        make_perfcheck.wait()

        tree = ET.parse(path_to_model)

        temp_df = get_implemented_conv(tree)
        if len(temp_df) != 0:
            print('\nAlready will be removed implemented:')
            print(get_implemented_conv(tree))
            for _index, row in temp_df.iterrows():
                revert_conv(tree, row)

        print('\nGetting current performance')
        name = 'old'
        currentTime, currentFPS, oldTypesTimes, oldStagesTimes = perfcheck(netName, name, tree, path_to_xml, image_dir, num_control_iters)
        while currentTime == 0:
            currentTime, currentFPS, oldTypesTimes, oldStagesTimes = perfcheck(netName, name, tree, path_to_xml, image_dir, num_control_iters)
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
            name = str(round(row['HW']))
            for _indexTry, rowTry in df_BetterTimes.iterrows():
                print('Try DimH='+str(rowTry['DimH'])+' DimW='+str(rowTry['DimW'])+
                    ' for InC='+str(rowTry['InC'])+' OutC='+str(rowTry['OutC']))
                implement_conv(tree, rowTry)
                tempTime, tempFPS, _tempTypesTimes, _tempStagesTimes = perfcheck(netName,
                    name+'-'+rowTry['Name']+'-'+str(round(rowTry['DimH']))+':'+str(round(rowTry['DimW'])), tree, path_to_xml, image_dir, num_iters)

                trys = 0
                while tempTime == 0 and trys < 10:
                    tempTime, tempFPS, _tempTypesTimes, _tempStagesTimes = perfcheck(netName,
                        name+'-'+rowTry['Name']+'-'+str(round(rowTry['DimH']))+':'+str(round(rowTry['DimW'])), tree, path_to_xml, image_dir, num_iters)
                    trys += 1

                if tempTime < bestTime:
                    bestTime = tempTime
                    bestFPS = tempFPS
                    bestRow = rowTry

                revert_conv(tree, rowTry)
            if bestTime < currentTime:
                implement_conv(tree, bestRow)
                currentTime = bestTime
                currentFPS = bestFPS
                print('Better reshape: DimH='+str(bestRow['DimH'])+' DimW='+str(bestRow['DimW']))
                print('Better Time: '+str(currentTime))
                print('Better FPS: '+str(currentFPS))
            else: print('No better reshape')

        name = 'old'
        print('\nOld Time: '+str(oldTime))
        print('Old FPS: '+str(oldFPS))
        oldStagesTimes.to_csv(os.path.join(openvino_path,
            'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/old_stages_times('+netName+'_'+name+').txt'), index=False)

        name = 'final'
        newTime, newFPS, newTypesTimes, newStagesTimes = perfcheck(netName, name, tree, path_to_xml, image_dir, num_iters)
        while newTime == 0:
            newTime, newFPS, newTypesTimes, newStagesTimes = perfcheck(netName, name, tree, path_to_xml, image_dir, num_iters)
        print('\nNew Time: '+str(newTime))
        print('New FPS: '+str(newFPS))
        newStagesTimes.to_csv(os.path.join(openvino_path,
            'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/new_stages_times('+netName+'_'+name+').txt'), index=False)

        def table_compare_types(oldTypesTimes, newTypesTimes):
            oldTypesTimes['Ver'] = 'Old'
            newTypesTimes['Ver'] = 'New'
            return pd.concat([oldTypesTimes, newTypesTimes])
        table_compare_types(oldTypesTimes, newTypesTimes).to_csv(os.path.join(openvino_path,
            'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/compare_types('+netName+'_'+name+').txt'), index=False)

        def table_compare_stages_times(oldStagesTimes, newStagesTimes):
            oldStagesTimes['Ver'] = 'Old'
            newStagesTimes['Ver'] = 'New'
            return pd.concat([oldStagesTimes, newStagesTimes])
        compareStages = table_compare_stages_times(oldStagesTimes, newStagesTimes)
        compareStages.to_csv(os.path.join(openvino_path,
            'cmake-build-for-improve_through_reshape/temp-files-for-improve_through_reshape/compare_stages('+netName+'_'+name+').txt'), index=False)

        print('\nNew reshape dims:')
        print(get_implemented_conv(tree))

    LongOptimize(df_AllConv1x1, df_AllTimeConv1x1, openvino_path, path_to_model, path_to_image_dir, num_iterations, num_control_iterations)

    end = datetime.now()

    print(start)
    print(end)
    print(end-start)
else:
    rxInputXML = r'\.xml'
    rxInputONNX = r'\.onnx'
    if re.search(rxInputXML, path_to_model) is None and re.search(rxInputONNX, path_to_model) is None:
        print('Path not to xml file or onnx file')
        exit()

    from openvino.inference_engine import IECore, IENetwork
    import ngraph as ng

    def get_net(model: str, core: IECore):
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = core.read_network(model=model_xml, weights=model_bin)
    return net
    def load_model(path_to_model: str):
    core = IECore()
    net = get_net(model=path_to_model, core=core)

    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()

    listOfConv = []

    for op in ops:
        if op.get_type_name() == 'Convolution':
            in_shapes = [sh for sh in op.input(0).get_shape()]
            out_shapes = [sh for sh in op.shape]
            attrs = op._get_attributes()
            pads_begin = attrs['pads_begin']
            pads_end = attrs['pads_end']
            if in_shapes[2:] == out_shapes[2:] and in_shapes[0] == out_shapes[0] == 1 and pads_begin == pads_end == [0,0]:
                name = op.get_name()
                dims = { 'DimH' : in_shapes[2], }
                rt_info = op.get_
                if rt_info["ConvReshape"] ==:
                    op.get_rt_info()["ConvReshape"] = "44"
                    if op.get_rt_info()["ConvReshape"] == "44":
                        print(op.get_rt_info())
                        print(op.get_rt_info()["ConvReshape"])
                        print(op.get_rt_info()["Variant::RuntimeAttribute::FusedNames"])

    #     net_graph_file_xml = os.path.expanduser('~') + '/Downloads/net_graph.xml'
    #     net_graph_file_bin = os.path.expanduser('~') + '/Downloads/net_graph.bin'
    #     net.serialize(net_graph_file_xml, net_graph_file_bin)

    #     print("sdssad")
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
    #                 if(i == 2):
    #                     op.get_rt_info()["ConvReshape"] = "66"
    #                     if op.get_rt_info()["ConvReshape"] == "66":
    #                         print(op.get_rt_info())
    #                         print(op.get_rt_info()["ConvReshape"])
    #                         print(op.get_rt_info()["Variant::RuntimeAttribute::FusedNames"])

    #     net_graph_file_xml = os.path.expanduser('~') + '/Downloads/net_graphTET.xml'
    #     net_graph_file_bin = os.path.expanduser('~') + '/Downloads/net_graphTET.bin'
    #     net.serialize(net_graph_file_xml, net_graph_file_bin)

    # load_model(path_to_model)
    def parse_xml(path_to_model: str):
        tree = ET.parse(path_to_model)
        root = tree.getroot()
        layers = root.find('layers')

        netName = root.get('name')
        listOfConv = []

        for layer in layers.iter('layer'):
            if layer.get('type') == 'Convolution':
                data = layer.find('data')

                strides = data.get('strides').split(',')
                strides = map(int, strides)
                strides = list(strides)
                if strides != [1,1]: continue

                input = layer.find('input')
                if input is None: continue

                flag_skip = False
                inC = 0
                outC = 0
                dimH = 0
                dimW = 0
                for port in input:
                    if port.get('id') == '0':
                        dims = list(port)

                        inC = int(dims[1].text)
                        dimH = int(dims[2].text)
                        dimW = int(dims[3].text)

                        if dims[0].text != '1' or dims[2].text == dims[3].text == '1':
                            flag_skip = True
                            break
                    elif port.get('id') == '1':
                        dims = list(port)

                        outC = int(dims[0].text)

                        if dims[2].text != '1' or dims[3].text != '1':
                            flag_skip = True
                            break
                if flag_skip: continue
                name = layer.get('name')

                currentDict = {'Name' : name, 'InC' : inC, 'OutC' : outC, 'DimH' : dimH, 'DimW' : dimW}
                listOfConv.append(currentDict) if currentDict not in listOfConv else listOfConv
        return netName, listOfConv

    print('Parse .xml')
    netName, listOfConv1x1 = parse_xml(path_to_model)

    print('\nAll convolutions 1x1 in net:')
    df_AllConv1x1 = pd.DataFrame(listOfConv1x1)
    print(df_AllConv1x1)

    print('\nDifferent convolutions 1x1 in net:')
    df_DiffConv1x1 = df_AllConv1x1.drop(columns=['Name']).drop_duplicates().reset_index(drop=True)
    print(df_DiffConv1x1)

print('Done')