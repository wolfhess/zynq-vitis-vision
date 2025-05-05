#include "xf_dense_npyr_optical_flow_accel_config.h"


typedef ap_axiu <AXIS_W, 1, 1, 1> stream_t;
typedef ap_axiu <32, 1, 1, 1> stream_out;

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
                hls::stream<stream_out>& AXI_video_strm) {
    stream_out axi;
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






void dense_non_pyr_of_accel(hls::stream<ap_axiu<AXIS_W,1,1,1> >& img_prev,
                            hls::stream<ap_axiu<AXIS_W,1,1,1> >& img_curr,
                            hls::stream<ap_axiu<32,1,1,1> >& img_outx,
                            hls::stream<ap_axiu<32,1,1,1> >& img_outy,
                            int rows,
                            int cols) {
    #pragma HLS INTERFACE axis     port=img_prev
    #pragma HLS INTERFACE axis     port=img_curr
    #pragma HLS INTERFACE axis     port=img_outx
    #pragma HLS INTERFACE axis     port=img_outy
    #pragma HLS INTERFACE s_axilite port=cols  
    #pragma HLS INTERFACE s_axilite port=rows  
    #pragma HLS INTERFACE s_axilite port=return

    xf::cv::Mat<IN_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX, XF_CV_DEPTH_IN_PREV> in_prev_mat(rows, cols);
    xf::cv::Mat<IN_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX, XF_CV_DEPTH_IN_CURR> in_curr_mat(rows, cols);
    xf::cv::Mat<OUT_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX, XF_CV_DEPTH_OUTX> outx_mat(rows, cols);
    xf::cv::Mat<OUT_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX, XF_CV_DEPTH_OUTY> outy_mat(rows, cols);

    #pragma HLS DATAFLOW

    axis2xfMat<AXIS_W, IN_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX>(img_prev, in_prev_mat);
    axis2xfMat<AXIS_W, IN_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX>(img_curr, in_curr_mat);
    
    xf::cv::DenseNonPyrLKOpticalFlow<KMED, IN_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX, XF_USE_URAM, XF_CV_DEPTH_IN_CURR,
                                     XF_CV_DEPTH_IN_PREV, XF_CV_DEPTH_OUTX, XF_CV_DEPTH_OUTY>(in_prev_mat, in_curr_mat,
                                                                                              outx_mat, outy_mat);
    xfMat2axis<32, OUT_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX>(outx_mat, img_outx);
    xfMat2axis<32, OUT_TYPE, MAX_HEIGHT, MAX_WIDTH, NPPCX>(outy_mat, img_outy);

}
