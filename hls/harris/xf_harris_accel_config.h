#ifndef _XF_HARRIS_CONFIG_H_
#define _XF_HARRIS_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "features/xf_harris.hpp"

#define WIDTH 1920
#define HEIGHT 1080
#define IMGSIZE WIDTH* HEIGHT
#define XF_CV_DEPTH_IN 2
#define XF_CV_DEPTH_OUT 2
#define MAXCORNERS 1024

#define FILTER_WIDTH 3
#define BLOCK_WIDTH 3
#define NMS_RADIUS 1
#define XF_USE_URAM 0

#define AXIS_W 64

#define NPPCX XF_NPPC8

#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1

#define CV_IN_TYPE CV_8UC1
#define CV_OUT_TYPE CV_8UC1


void cornerHarris_accel(
    hls::stream<ap_axiu<AXIS_W,1,1,1> >& img_inp, hls::stream<ap_axiu<AXIS_W,1,1,1> >&  img_out, int rows, int cols, int threshold, int k);
#endif
