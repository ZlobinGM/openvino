// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This is the pass that finds the pattern which consists of a Convolution stage
// with 2 consumers - Power and Concat stages (Power stage is also followed by Concat),
// ScaleShift (or Scale, it depends on biases) which comes after Concat,
// and the last one is Relu.

#include <vpu/middleend/pass_manager.hpp>

// #include <vpu/middleend/hw/conv_tiling/hw_stage_tiler.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(reshapeTiles);
    for (const auto& stage : model->getStages()) {

    }
}

}  // namespace

Pass::Ptr PassManager::reshapeTiles() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
