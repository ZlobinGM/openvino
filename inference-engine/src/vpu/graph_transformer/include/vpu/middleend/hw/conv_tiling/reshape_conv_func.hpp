// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This pass changes geometry of convolution stages in order
// to get more efficient HW tiling (pass "hwConvTiling") using reshape stages.

int ChoiceDimH(int InC, int OutC, int DimH, int DimW) {
    // if (DimH * DimW == 676 &&
    //     ((InC == 256 && OutC == 128) ||
    //     (InC == 512 && OutC == 255) ||
    //     (InC == 512 && OutC == 256) ||
    //     (InC == 768 && OutC == 256))) {
    //     return 52;
    // }
    if (DimH * DimW == 2704 &&
        ((InC == 256 && OutC == 128) ||
        (InC == 256 && OutC == 255) ||
        (InC == 384 && OutC == 128)))
        return 169;
    // if (DimH * DimW == 10816 &&
    //     (InC == 128 && OutC == 64))
    //         return 169;
    // if (DimH * DimW == 43264 &&
    //     (InC == 64 && OutC == 32))
    //     return 338;
    return 0;
}