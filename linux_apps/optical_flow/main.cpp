#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/time.h>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "axi_dma.h"


#define OF_AP_CTRL_REG     0x00
#define OF_GIE_REG         0x04
#define OF_IER_REG         0x08
#define OF_ISR_REG         0x0C
#define OF_ROWS_REG        0x10
#define OF_COLS_REG        0x18

#define AP_START_MASK      (1 << 0)
#define AP_DONE_MASK       (1 << 1)
#define AP_IDLE_MASK       (1 << 2)
#define AP_READY_MASK      (1 << 3)
#define AUTO_RESTART_MASK  (1 << 7)



cv::Mat draw_optical_flow(cv::Mat flow, int stride = 20) {
	cv::Mat vec_frame = cv::Mat::zeros(flow.size(), CV_8UC1);
	double max_norm = 0;
	for (int y = 0; y < flow.rows; y++) {
		for (int x = 0; x < flow.cols; x++) {
			cv::Vec2f vec = flow.at<cv::Vec2f>(y, x);
			double norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
			if (norm > max_norm)
				max_norm = norm;

		}
	}

	for (int y = 0; y < flow.rows; y += stride) {
		for (int x = 0; x < flow.cols; x += stride) {
			cv::Vec2f vec = flow.at<cv::Vec2f>(y, x);
			double angle = std::atan2(vec[1], vec[0]);
			double norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
			double norm_vec_length = (norm * stride) / max_norm;
			cv::Point start(x, y);
			cv::Point end(
				static_cast<int>(x + norm_vec_length * std::cos(angle)),
				static_cast<int>(y + norm_vec_length * std::sin(angle))
			);
			cv::line(vec_frame, start, end, cv::Scalar(255), 1, cv::LINE_AA);
			cv::circle(vec_frame, end, 2, cv::Scalar(255), -1, cv::LINE_AA);
		}
	}
	cvtColor(vec_frame, vec_frame, cv::COLOR_GRAY2BGR);
	return vec_frame;
}



cv::Mat draw_optical_dense_flow(cv::Mat flow) {
	cv::Mat flow_parts[2];
	cv::split(flow, flow_parts);
	cv::Mat magnitude, angle, magn_norm;
	cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
	cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
	angle *= ((1.f / 360.f) * (180.f / 255.f));
	//build hsv image
	cv::Mat _hsv[3], hsv, hsv8, bgr;
	_hsv[0] = angle;
	_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
	_hsv[2] = magn_norm;
	cv::merge(_hsv, 3, hsv);
	hsv.convertTo(hsv8, CV_8U, 255.0);
	cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
	return bgr;
}

double computeAEE(const cv::Mat& flow_pred, const cv::Mat& flow_gt) {
    CV_Assert(flow_pred.size() == flow_gt.size());
    CV_Assert(flow_pred.type() == CV_32FC2 && flow_gt.type() == CV_32FC2);

    cv::Mat diff;
    cv::subtract(flow_pred, flow_gt, diff);
    std::vector<cv::Mat> channels(2);
    cv::split(diff, channels);  // channels[0] = dx, channels[1] = dy
    cv::Mat error_magnitude;
    cv::magnitude(channels[0], channels[1], error_magnitude);  // тип CV_32F
    cv::Scalar mean_error = cv::mean(error_magnitude);

    return mean_error[0];
}


std::pair<double, double> computeMSE_XY(const cv::Mat& flow_pred, const cv::Mat& flow_gt) {
    CV_Assert(flow_pred.size() == flow_gt.size());
    CV_Assert(flow_pred.type() == CV_32FC2 && flow_gt.type() == CV_32FC2);

    std::vector<cv::Mat> pred_channels(2), gt_channels(2);
    cv::split(flow_pred, pred_channels);  // pred_channels[0] = X, [1] = Y
    cv::split(flow_gt, gt_channels);

    cv::Mat diff_x = pred_channels[0] - gt_channels[0];
    cv::Mat diff_y = pred_channels[1] - gt_channels[1];

    cv::Mat sqr_x, sqr_y;
    cv::multiply(diff_x, diff_x, sqr_x);
    cv::multiply(diff_y, diff_y, sqr_y);

    double mse_x = cv::mean(sqr_x)[0];
    double mse_y = cv::mean(sqr_y)[0];

    return {mse_x, mse_y};
}





void of_read_registers(volatile unsigned int *of_ip)
{
    unsigned int ap_ctrl_val = of_ip[OF_AP_CTRL_REG >> 2];
    printf("ap_control (offset 0x00) = 0x%08X\n", ap_ctrl_val);
    printf("  ap_start     = %d\n", (ap_ctrl_val & AP_START_MASK) ? 1 : 0);
    printf("  ap_done      = %d\n", (ap_ctrl_val & AP_DONE_MASK)  ? 1 : 0);
    printf("  ap_idle      = %d\n", (ap_ctrl_val & AP_IDLE_MASK)  ? 1 : 0);
    printf("  ap_ready     = %d\n", (ap_ctrl_val & AP_READY_MASK) ? 1 : 0);
    printf("  auto_restart = %d\n", (ap_ctrl_val & AUTO_RESTART_MASK) ? 1 : 0);

    unsigned int gier_val = of_ip[OF_GIE_REG >> 2];
    unsigned int ier_val  = of_ip[OF_IER_REG >> 2];
    unsigned int isr_val  = of_ip[OF_ISR_REG >> 2];
    printf("GIER (offset 0x04) = 0x%08X\n", gier_val);
    printf("IER  (offset 0x08) = 0x%08X\n", ier_val);
    printf("ISR  (offset 0x0C) = 0x%08X\n", isr_val);

    unsigned int rows_val = of_ip[OF_ROWS_REG >> 2];
    unsigned int cols_val = of_ip[OF_COLS_REG >> 2];
    printf("rows (offset 0x10) = %u\n", rows_val);
    printf("cols (offset 0x18) = %u\n", cols_val);
    printf("------------------------------------------------\n");
}

int main(int argc, char** argv)
{

    int width  = 1920;
    int height = 1080;
    int img_size = width * height;
    cv::Mat previous, current;
    cv::Mat frame, gray, resized, flow, flow_fpga;
    int frameCount = 0;
    double AEE = 0;
    double MSE_X = 0;
    double MSE_Y = 0;
    double totalTimeMs_hard = 0;
    double totalTimeMs_soft = 0;


    if (argc < 2) {
        printf("Usage: %s <path_to_video>\n", argv[0]);
        return -1;
    }


    const char* videoPath = argv[1];
	cv::VideoCapture cap(videoPath);

	if (!cap.isOpened()) {
		std::cerr << "Error: Cannot open video: " << videoPath << std::endl;
		return 1;
	}



	int ddr_memory = open("/dev/mem", O_RDWR | O_SYNC);

	off_t phys_addr_dma1 = 0x40400000;
	off_t phys_addr_dma2 = 0x41E00000;
	size_t map_size = 65536;

	unsigned int *dma1_virtual_addr = (unsigned int *)mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, phys_addr_dma1);
	unsigned int *dma2_virtual_addr = (unsigned int *)mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, phys_addr_dma2);
	unsigned int *virtual_src_addr1 = (unsigned int *)mmap(NULL, img_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x0e000000);
	unsigned int *virtual_src_addr2 = (unsigned int *)mmap(NULL, img_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x0f000000);
	unsigned int *virtual_dst_addr1 = (unsigned int *)mmap(NULL, img_size*4, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x10000000);
	unsigned int *virtual_dst_addr2 = (unsigned int *)mmap(NULL, img_size*4, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x11000000);

	unsigned int *of_virtual_addr = (unsigned int *)mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory,0x40000000);



	of_virtual_addr[OF_ROWS_REG >> 2] = (unsigned int)height;
	of_virtual_addr[OF_COLS_REG >> 2] = (unsigned int)width;

    of_read_registers(of_virtual_addr);



    unsigned int ap_ctrl_val = of_virtual_addr[OF_AP_CTRL_REG >> 2];
    ap_ctrl_val |= AUTO_RESTART_MASK;
    ap_ctrl_val |= AP_START_MASK;
    of_virtual_addr[OF_AP_CTRL_REG >> 2] = ap_ctrl_val;


	cv::Mat frame0(height, width, CV_8UC1, virtual_src_addr1);
	cv::Mat frame1(height, width, CV_8UC1, virtual_src_addr2);
	static std::vector<float> buf_flowx(width * height);
	static std::vector<float> buf_flowy(width * height);
	cv::Mat flowx(height, width, CV_32FC1, buf_flowx.data());
	cv::Mat flowy(height, width, CV_32FC1, buf_flowy.data());
	printf("float32: %i\n", sizeof(float));

    write_dma(dma1_virtual_addr, S2MM_CONTROL_REGISTER, RESET_DMA);
    write_dma(dma1_virtual_addr, MM2S_CONTROL_REGISTER, RESET_DMA);
    write_dma(dma2_virtual_addr, S2MM_CONTROL_REGISTER, RESET_DMA);
    write_dma(dma2_virtual_addr, MM2S_CONTROL_REGISTER, RESET_DMA);


    write_dma(dma1_virtual_addr, S2MM_CONTROL_REGISTER, HALT_DMA);
    write_dma(dma1_virtual_addr, MM2S_CONTROL_REGISTER, HALT_DMA);
    write_dma(dma2_virtual_addr, S2MM_CONTROL_REGISTER, HALT_DMA);
    write_dma(dma2_virtual_addr, MM2S_CONTROL_REGISTER, HALT_DMA);


    write_dma(dma1_virtual_addr, S2MM_CONTROL_REGISTER, ENABLE_ALL_IRQ);
    write_dma(dma1_virtual_addr, MM2S_CONTROL_REGISTER, ENABLE_ALL_IRQ);
    write_dma(dma2_virtual_addr, S2MM_CONTROL_REGISTER, ENABLE_ALL_IRQ);
    write_dma(dma2_virtual_addr, MM2S_CONTROL_REGISTER, ENABLE_ALL_IRQ);


    write_dma(dma1_virtual_addr, MM2S_SRC_ADDRESS_REGISTER, 0x0e000000);
    write_dma(dma1_virtual_addr, S2MM_DST_ADDRESS_REGISTER, 0x10000000);
    write_dma(dma2_virtual_addr, MM2S_SRC_ADDRESS_REGISTER, 0x0f000000);
    write_dma(dma2_virtual_addr, S2MM_DST_ADDRESS_REGISTER, 0x11000000);


    write_dma(dma1_virtual_addr, MM2S_CONTROL_REGISTER, RUN_DMA);
    write_dma(dma1_virtual_addr, S2MM_CONTROL_REGISTER, RUN_DMA);
    write_dma(dma2_virtual_addr, MM2S_CONTROL_REGISTER, RUN_DMA);
    write_dma(dma2_virtual_addr, S2MM_CONTROL_REGISTER, RUN_DMA);

    write_dma(dma1_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, img_size);
    write_dma(dma1_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, img_size*4);
    write_dma(dma2_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, img_size);
    write_dma(dma2_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, img_size*4);

    //usleep(100 * 1000);


    while (true) {
    		if (!cap.read(frame)) {
    			std::cout << "End of video or failed to read frame." << std::endl;
    			break;
    		}

    		if (frameCount >= 2)
    			break;

    		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    		if (frameCount == 0){
				cv::resize(gray, frame0, cv::Size(1920, 1080));
    			frameCount++;
    			continue;
    		}

    		cv::resize(gray, frame1, cv::Size(width, height));



    		//___________soft__________
    		int64 t0_soft = cv::getTickCount();

    	    cv::calcOpticalFlowFarneback(
				frame0, frame1, flow,
    	        0.5,    // pyramid scale
    	        1,      // levels = 1
    	        25,      // window size
    	        1,      // iterations
    	        3,      // poly_n
    	        1.1,    // poly_sigma
    	        0
    	    );

    	    int64 t1_soft = cv::getTickCount();

    		totalTimeMs_soft += (t1_soft - t0_soft) / cv::getTickFrequency();
    		//_________________________



    		//___________fpga__________
    		int64 t0_hard = cv::getTickCount();

    		msync(virtual_src_addr1, img_size, MS_SYNC);
			msync(virtual_src_addr2, img_size, MS_SYNC);

		    write_dma(dma1_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, img_size);
		    write_dma(dma1_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, img_size*4);
		    write_dma(dma2_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, img_size);
		    write_dma(dma2_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, img_size*4);

			dma_mm2s_sync(dma1_virtual_addr);
			dma_mm2s_sync(dma2_virtual_addr);
			dma_s2mm_sync(dma1_virtual_addr);
			dma_s2mm_sync(dma2_virtual_addr);

			memcpy(buf_flowx.data(), virtual_dst_addr1, img_size*4);
			memcpy(buf_flowy.data(), virtual_dst_addr2, img_size*4);

    		int64 t1_hard = cv::getTickCount();

    		totalTimeMs_hard += (t1_hard - t0_hard) / cv::getTickFrequency();
    		// _________________________

    		frame1.copyTo(frame0);

    		std::vector<cv::Mat> fpga_channels = {flowx, flowy};
			cv::merge(fpga_channels, flow_fpga);

		    AEE += computeAEE(flow_fpga, flow);
		    auto [mse_x, mse_y] = computeMSE_XY(flow_fpga, flow);
		    MSE_X += mse_x;
		    MSE_Y += mse_y;
    		printf("frame finished\n");

    		frameCount++;


    	}
	cap.release();

    printf("AEE: %f\n", AEE / (frameCount-1));
    printf("MSE X: %f\n", MSE_X / (frameCount-1));
    printf("MSE Y: %f\n", MSE_Y / (frameCount-1));

    cv::imwrite("fpga_of.png", draw_optical_dense_flow(flow_fpga));
    cv::imwrite("ocv_of.png", draw_optical_dense_flow(flow));


	double avgMs_hard = 1000.0 * totalTimeMs_hard / (frameCount - 1);
	printf("Average processing time fpga per frame: %.3f ms\n", avgMs_hard);

	double avgMs_soft = 1000.0 * totalTimeMs_soft / (frameCount - 1);
	printf("Average processing time ocv per frame: %.3f ms\n", avgMs_soft);


	munmap((void*)of_virtual_addr, 65536);
	munmap((void*)dma1_virtual_addr, 65536);
	munmap((void*)dma2_virtual_addr, 65536);
	munmap((void*)virtual_src_addr1, img_size);
	munmap((void*)virtual_src_addr2, img_size);
	munmap((void*)virtual_dst_addr1, img_size*4);
	munmap((void*)virtual_dst_addr2, img_size*4);
	close(ddr_memory);

	printf("\n");

    return 0;
}
