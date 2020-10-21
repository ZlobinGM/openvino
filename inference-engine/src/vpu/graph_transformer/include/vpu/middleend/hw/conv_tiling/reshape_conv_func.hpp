// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This pass changes geometry of convolution stages in order
// to get more efficient HW tiling (pass "hwConvTiling") using reshape stages.

int ChoiceDimH(std::string Name, int InC, int OutC, int DimH, int DimW) {
    // if (DimH * DimW == 676 &&
    //     ((InC == 256 && OutC == 128) ||
    //     (InC == 512 && OutC == 255) ||
    //     (InC == 512 && OutC == 256) ||
    //     (InC == 768 && OutC == 256))) {
    //     return 13;
    // }
    // if (DimH * DimW == 2704 &&
    //     ((InC == 256 && OutC == 128) ||
    //     (InC == 256 && OutC == 255) ||
    //     (InC == 384 && OutC == 128)))
    //     return 169;
    // if (DimH * DimW == 10816 &&
    //     (InC == 128 && OutC == 64))
    //         return 169;
    // if (DimH * DimW == 43264 &&
    //     (InC == 64 && OutC == 32))
    //     return 338;
    // person-detection-action-recognition-0006 100    299.55 ms -> 296.173 ms | 6.5794 fps -> 6.67541 fps
    // if ((DimH * DimW == 68000) && (InC == 32) && (OutC == 8)) return 425;
    // if ((DimH * DimW == 1075) && (InC == 64) && (OutC == 256)) return 5;
    // mobilenet-v2 100    no better
    // yolo_v3 100    408.958 ms -> 396.079 ms | 4.8594 fps -> 5.01494 fps
    // if ((DimH * DimW == 2704) && (InC == 256) && (OutC == 128)) return 169;
    // if ((DimH * DimW == 676) && (InC == 768) && (OutC == 256)) return 13;
    // prnet 100    319.006 ms -> 318.327 ms | 6.26775 fps -> 6.25418 fps
    // if ((DimH * DimW == 64) && (InC == 512) && (OutC == 256)) return 2;
    // product-detection-0001 100    123.046 ms -> 118.789 ms | 15.8654 fps -> 16.3585 fps
    // if ((DimH * DimW == 16384) && (InC == 144) && (OutC == 24)) return 8;
    // if ((DimH * DimW == 1024) && (InC == 576) && (OutC == 96)) return 64;
    // if ((DimH * DimW == 256) && (InC == 576) && (OutC == 160)) return 64;
    // if ((DimH * DimW == 256) && (InC == 1280) && (OutC == 256)) return 1;
    // if ((DimH * DimW == 64) && (InC == 256) && (OutC == 512)) return 16;
    // if ((DimH * DimW == 64) && (InC == 512) && (OutC == 24)) return 16;
    // se-inception 100   no better
    // text-detection-0003
    // if ((DimH * DimW == 245760) && (InC == 32) && (OutC == 32)) return 640;
    // if ((DimH * DimW == 245760) && (InC == 16) && (OutC == 96)) return 64;
    // if ((DimH * DimW == 15360) && (InC == 144) && (OutC == 32)) return 16;
    // if ((DimH * DimW == 15360) && (InC == 192) && (OutC == 64)) return 192;
    // if ((DimH * DimW == 15360) && (InC == 384) && (OutC == 384)) return 6;
    // if ((DimH * DimW == 15360) && (InC == 384) && (OutC == 64)) return 32;
    // if ((DimH * DimW == 3840) && (InC == 384) && (OutC == 384)) return 120;
    // if ((DimH * DimW == 3840) && (InC == 384) && (OutC == 96)) return 3;
    // if ((DimH * DimW == 3840) && (InC == 576) && (OutC == 576)) return 3;
    // if ((DimH * DimW == 3840) && (InC == 96) && (OutC == 16)) return 320;
    // if ((DimH * DimW == 960) && (InC == 576) && (OutC == 576)) return 48;
    // if ((DimH * DimW == 960) && (InC == 576) && (OutC == 160)) return 32;
    // if ((DimH * DimW == 960) && (InC == 960) && (OutC == 960)) return 10;
    // if ((DimH * DimW == 960) && (InC == 960) && (OutC == 160)) return 40;
    // if ((DimH * DimW == 960) && (InC == 960) && (OutC == 320)) return 60;
    // googlenet-v4 100    213.157 ms -> 166.39 ms | 9.82335 fps -> 11.8762 fps
    // if ((DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // googlenet-v4 100
    // if ((Name == "inception_b1_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b1_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b1_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b2_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b2_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b3_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b3_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b3_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b4_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b4_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b5_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b5_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b5_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b6_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b6_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b6_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b7_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b7_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b7_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "reduction_b_3x3_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "reduction_b_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 256)) return 1;
    // if ((Name == "inception_c1_1x1_4_scale/Add_") && (DimH * DimW == 64) && (InC == 1536) && (OutC == 384)) return 1;
    // if ((Name == "inception_c1_1x1_2_scale/Add_") && (DimH * DimW == 64) && (InC == 1536) && (OutC == 256)) return 1;
    // if ((Name == "inception_b7_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b7_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // googlenet-v4 if cond
    // if ((Name == "inception_b1_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b1_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b2_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b2_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b2_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b3_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b3_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b4_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b4_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b5_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b5_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b6_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b6_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b6_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b7_1x1_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 128)) return 1;
    // if ((Name == "inception_b7_7x1_2_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "inception_b7_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 192)) return 1;
    // if ((Name == "reduction_b_1x7_reduce_scale/Add_") && (DimH * DimW == 289) && (InC == 1024) && (OutC == 256)) return 1;
    // if ((Name == "inception_c1_1x1_4_scale/Add_") && (DimH * DimW == 64) && (InC == 1536) && (OutC == 384)) return 1;
    // if ((Name == "inception_c2_1x1_4_scale/Add_") && (DimH * DimW == 64) && (InC == 1536) && (OutC == 384)) return 1;
    // if ((Name == "inception_c2_1x1_2_scale/Add_") && (DimH * DimW == 64) && (InC == 1536) && (OutC == 256)) return 1;
    // if ((Name == "inception_c3_1x1_2_scale/Add_") && (DimH * DimW == 64) && (InC == 1536) && (OutC == 256)) return 1;
    return 0;
}