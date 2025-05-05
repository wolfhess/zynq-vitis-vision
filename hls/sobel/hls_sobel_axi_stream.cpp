#include "hls_sobel_axi_stream.hpp"
#include "imgproc/xf_sobel.hpp"
#include "imgproc/xf_add_weighted.hpp"

typedef ap_axiu <AXIS_W, 1, 1, 1> stream_t;

template <int W, int TYPE, int ROWS, int COLS, int NPPC>
void axis2xfMat(hls::stream<stream_t>& AXI_video_strm,
               xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& img) {
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


template <int W, int TYPE, int ROWS, int COLS, int NPPC>
void xfMat2axis(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& img,
               hls::stream<stream_t>& AXI_video_strm) {
    stream_t axi;
    const int PPC = 1 << XF_BITSHIFT(NPPC);
    int rows = img.rows;
    int cols = img.cols >> XF_BITSHIFT(NPPC);

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


void hls_sobel_axi_stream_top(
                hls::stream<ap_axiu<AXIS_W,1,1,1> >& _src,
                hls::stream<ap_axiu<AXIS_W,1,1,1> >& _dst,
                int rows,
                int cols) {

    #pragma HLS INTERFACE axis port=_src
    #pragma HLS INTERFACE axis port=_dst
    #pragma HLS INTERFACE s_axilite port=rows           bundle=control
    #pragma HLS INTERFACE s_axilite port=cols           bundle=control
    #pragma HLS INTERFACE s_axilite port=return         bundle=control

    xf::cv::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8> img_buf_0(rows, cols);
    xf::cv::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8> img_buf_1a(rows, cols);
    xf::cv::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8> img_buf_1b(rows, cols);
    xf::cv::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8> img_buf_2(rows, cols);

    #pragma HLS dataflow

    axis2xfMat<AXIS_W, XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8>(_src, img_buf_0);
    xf::cv::Sobel<0, 3, XF_8UC1, XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8, false>(img_buf_0, img_buf_1a, img_buf_1b);
    xf::cv::addWeighted<XF_8UC1, XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8>(img_buf_1a, 0.5, img_buf_1b, 0.5, 0, img_buf_2);
    xfMat2axis<AXIS_W, XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC8>(img_buf_2, _dst);

    return;
}
