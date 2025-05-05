#include "xf_harris_accel_config.h"

typedef ap_axiu <AXIS_W, 1, 1, 1> stream_t;

template<int W, int TYPE, int ROWS, int COLS, int NPPC, int DEPTH>
void axis2xfMat(hls::stream<stream_t>& AXI_video_strm,
                xf::cv::Mat<TYPE, ROWS, COLS, NPPC, DEPTH>& img) {
    stream_t axi;
    const int PPC = 1 << XF_BITSHIFT(NPPC);
    int rows = img.rows;
    int cols = img.cols >> XF_BITSHIFT(NPPC);

row_loop:
    for (int i = 0; i < rows; i++) {
    col_loop:
        for (int j = 0; j < cols; j++) {
            #pragma HLS PIPELINE II=1
            AXI_video_strm.read(axi);
            img.write(i * cols + j, axi.data);
        }
    }
}



template<int W, int TYPE, int ROWS, int COLS, int NPPC, int DEPTH>
void xfMat2axis(xf::cv::Mat<TYPE, ROWS, COLS, NPPC, DEPTH>& img,
                hls::stream<stream_t>& AXI_video_strm) {
    stream_t axi;
    const int PPC = 1 << XF_BITSHIFT(NPPC); // 8
    int rows = img.rows;
    int cols = img.cols >> XF_BITSHIFT(NPPC); // cols = width/8

rowloop:
    for (int i = 0; i < rows; i++) {
    col_loop:
        for (int j = 0; j < cols; j++) {
            #pragma HLS PIPELINE II=1
            axi.data = img.read(i * cols + j);
            axi.keep = -1;
            axi.strb = -1;
            axi.last = (i == rows-1) && (j == cols-1);
            AXI_video_strm.write(axi);
        }
    }
}


void cornerHarris_accel(
    hls::stream<ap_axiu<AXIS_W,1,1,1> >& img_inp, hls::stream<ap_axiu<AXIS_W,1,1,1> >&  img_out, int rows, int cols, int threshold, int k) {
    #pragma HLS INTERFACE axis      port=img_inp
    #pragma HLS INTERFACE axis      port=img_out
   
    #pragma HLS INTERFACE s_axilite port=rows     
    #pragma HLS INTERFACE s_axilite port=cols     
    #pragma HLS INTERFACE s_axilite port=threshold     
    #pragma HLS INTERFACE s_axilite port=k     
    #pragma HLS INTERFACE s_axilite port=return

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_IN> in_mat(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_OUT> out_mat(rows, cols);

    #pragma HLS DATAFLOW
    axis2xfMat<AXIS_W, IN_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_IN>(img_inp, in_mat);
    xf::cv::cornerHarris<FILTER_WIDTH, BLOCK_WIDTH, NMS_RADIUS, IN_TYPE, HEIGHT, WIDTH, NPPCX, XF_USE_URAM,
                         XF_CV_DEPTH_IN, XF_CV_DEPTH_OUT>(in_mat, out_mat, threshold, k);
    xfMat2axis<AXIS_W, OUT_TYPE, HEIGHT, WIDTH, NPPCX, XF_CV_DEPTH_OUT>(out_mat, img_out);
}
