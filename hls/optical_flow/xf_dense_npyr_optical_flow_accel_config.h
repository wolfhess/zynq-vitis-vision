#ifndef __XF_DENSE_NONPYR_OPTICAL_FLOW_CONFIG__
#define __XF_DENSE_NONPYR_OPTICAL_FLOW_CONFIG__

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "video/xf_dense_npyr_optical_flow.hpp"

#define AXIS_W 8

#define HEIGHT 1080
#define WIDTH 1920

#define XF_CV_DEPTH_IN_CURR 2
#define XF_CV_DEPTH_IN_PREV 2
#define XF_CV_DEPTH_OUTX 2
#define XF_CV_DEPTH_OUTY 2

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920
#define KMED 25

#define NPPCX XF_NPPC1

#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_32FC1

#define XF_USE_URAM false

void dense_non_pyr_of_accel(hls::stream<ap_axiu<AXIS_W,1,1,1> >& img_prev,
                            hls::stream<ap_axiu<AXIS_W,1,1,1> >& img_curr,
                            hls::stream<ap_axiu<32,1,1,1> >& img_outx,
                            hls::stream<ap_axiu<32,1,1,1> >& img_outy,
                            int rows,
                            int cols);

#endif
