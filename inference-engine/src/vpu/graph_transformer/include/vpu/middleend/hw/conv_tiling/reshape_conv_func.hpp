// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This pass changes geometry of convolution stages in order
// to get more efficient HW tiling (pass "hwConvTiling") using reshape stages.

int ChoiceDimH(int InC, int OutC, int DimH, int DimW) {
    return 0;
}