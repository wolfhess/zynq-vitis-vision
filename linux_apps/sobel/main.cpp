#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/time.h>
#include <dirent.h>
#include <sys/stat.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "axi_dma.h"

#define SOBEL_AP_CTRL_REG     0x00
#define SOBEL_GIE_REG         0x04
#define SOBEL_IER_REG         0x08
#define SOBEL_ISR_REG         0x0C
#define SOBEL_ROWS_REG        0x10
#define SOBEL_COLS_REG        0x18

#define AP_START_MASK      (1 << 0)
#define AP_DONE_MASK       (1 << 1)
#define AP_IDLE_MASK       (1 << 2)
#define AP_READY_MASK      (1 << 3)
#define AUTO_RESTART_MASK  (1 << 7)


using namespace std;


double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    // Проверка, что изображения имеют одинаковый размер и тип
    if (I1.size() != I2.size() || I1.type() != I2.type())
    {
        cerr << "err" << endl;
        return -1.0;
    }

    cv::Mat s1;
    cv::absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // преобразуем в 32-битный формат для вычислений
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = sum(s1);        // суммируем элементы по каналам

    // Для изображений с одним каналом используется только s.val[0]
    double sse = s.val[0];     // для одно-канальных изображений только первый элемент

    if( sse <= 1e-10) // для малых значений возвращаем 0
        return 0;
    else
    {
        double mse = sse / (double)(I1.total());  // Среднеквадратичная ошибка (MSE)
        double psnr = 10.0 * log10((255 * 255) / mse);  // PSNR в децибелах
        return psnr;
    }
}



double computeMSE(const cv::Mat& img1, const cv::Mat& img2) {
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    cv::Mat diff;
    cv::absdiff(img1, img2, diff);         // |img1 - img2|
    diff.convertTo(diff, CV_32F);          // преобразуем в float
    diff = diff.mul(diff);                 // (img1 - img2)^2

    double mse = cv::sum(diff)[0] / (img1.total());  // среднее значение
    return mse;
}




double computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    const double C1 = 6.5025, C2 = 58.5225;  // значения по умолчанию

    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;

    cv::GaussianBlur(img1f.mul(img1f), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    cv::GaussianBlur(img2f.mul(img2f), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    cv::GaussianBlur(img1f.mul(img2f), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat numerator = t1.mul(t2);

    cv::Mat t3 = mu1_sq + mu2_sq + C1;
    cv::Mat t4 = sigma1_sq + sigma2_sq + C2;
    cv::Mat denominator = t3.mul(t4);

    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);

    return mssim[0];  // для одноканального изображения
}





void sobel_read_registers(volatile unsigned int *sobel_ip)
{
    unsigned int ap_ctrl_val = sobel_ip[SOBEL_AP_CTRL_REG >> 2];
    printf("ap_control (offset 0x00) = 0x%08X\n", ap_ctrl_val);
    printf("  ap_start     = %d\n", (ap_ctrl_val & AP_START_MASK) ? 1 : 0);
    printf("  ap_done      = %d\n", (ap_ctrl_val & AP_DONE_MASK)  ? 1 : 0);
    printf("  ap_idle      = %d\n", (ap_ctrl_val & AP_IDLE_MASK)  ? 1 : 0);
    printf("  ap_ready     = %d\n", (ap_ctrl_val & AP_READY_MASK) ? 1 : 0);
    printf("  auto_restart = %d\n", (ap_ctrl_val & AUTO_RESTART_MASK) ? 1 : 0);

    unsigned int gier_val = sobel_ip[SOBEL_GIE_REG >> 2];
    unsigned int ier_val  = sobel_ip[SOBEL_IER_REG >> 2];
    unsigned int isr_val  = sobel_ip[SOBEL_ISR_REG >> 2];
    printf("GIER (offset 0x04) = 0x%08X\n", gier_val);
    printf("IER  (offset 0x08) = 0x%08X\n", ier_val);
    printf("ISR  (offset 0x0C) = 0x%08X\n", isr_val);

    unsigned int rows_val = sobel_ip[SOBEL_ROWS_REG >> 2];
    unsigned int cols_val = sobel_ip[SOBEL_COLS_REG >> 2];
    printf("rows (offset 0x10) = %u\n", rows_val);
    printf("cols (offset 0x18) = %u\n", cols_val);
    printf("------------------------------------------------\n");
}

int main(int argc, char** argv)
{
    int width  = 1920;
    int height = 1080;
    int img_size = width * height;
    double PSNR = 0;
    double MSE = 0;
    double SSIM = 0;

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

	off_t phys_addr = 0x40400000;
	size_t map_size = 65536;
	unsigned int *dma_virtual_addr = (unsigned int *)mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, phys_addr);

	if (dma_virtual_addr == MAP_FAILED) {
	    perror("mmap failed");
	    exit(EXIT_FAILURE);
	}

	unsigned int *virtual_src_addr = (unsigned int *)mmap(NULL, img_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x0e000000);
	unsigned int *virtual_dst_addr = (unsigned int *)mmap(NULL, img_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x0f000000);
	unsigned int *sobel_virtual_addr = (unsigned int *)mmap(NULL,map_size, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, 0x40000000);


    //memcpy(virtual_src_addr, input_img.data, img_size);
    //memset(virtual_dst_addr, 0, img_size);


	sobel_virtual_addr[SOBEL_ROWS_REG >> 2] = (unsigned int)height;
	sobel_virtual_addr[SOBEL_COLS_REG >> 2] = (unsigned int)width;

    //printf("Sobel IP regs:\n");
    //sobel_read_registers(sobel_virtual_addr);


	//memset(virtual_dst_addr, 0, img_size);
	//unsigned char *output_img = (unsigned char *)malloc(img_size);

    //struct timeval start, end;
   // double elapsed_time_ms;


    unsigned int ap_ctrl_val = sobel_virtual_addr[SOBEL_AP_CTRL_REG >> 2];
    ap_ctrl_val |= AUTO_RESTART_MASK;
    ap_ctrl_val |= AP_START_MASK;
    sobel_virtual_addr[SOBEL_AP_CTRL_REG >> 2] = ap_ctrl_val;



    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, RESET_DMA);
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, RESET_DMA);

    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, HALT_DMA);
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, HALT_DMA);

    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, ENABLE_ALL_IRQ);
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, ENABLE_ALL_IRQ);

    write_dma(dma_virtual_addr, MM2S_SRC_ADDRESS_REGISTER, 0x0e000000);

    write_dma(dma_virtual_addr, S2MM_DST_ADDRESS_REGISTER, 0x0f000000);

    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, RUN_DMA);

    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, RUN_DMA);

    write_dma(dma_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, img_size);

    write_dma(dma_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, img_size);

    usleep(100 * 1000);


    int frameCount = 0;
    double totalTimeMs = 0;
    double totalTimeMs_hard = 0;



	static std::vector<unsigned char> fast_buf(width * height);
	cv::Mat fast_mat(height, width, CV_8UC1, fast_buf.data());

    cv::Mat resized(height, width, CV_8UC1, virtual_src_addr);
    //cv::Mat fpga_result(height, width, CV_8UC1, virtual_dst_addr);
	cv::Mat frame, gray, sobel_x, sobel_y, ocv_result, fpga_color_soft;
	cv::Mat fpga_color, goy;

	while (true) {
		if (!cap.read(frame)) {
			std::cout << "End of video or failed to read frame." << std::endl;
			break;
		}

		if (frameCount >= 100) break;


		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::resize(gray, resized, resized.size());

		//___________soft ocv_________
		int64 t0 = cv::getTickCount();

		cv::Sobel(resized, sobel_x, CV_8U, 1, 0, 3);
		cv::Sobel(resized, sobel_y, CV_8U, 0, 1, 3);
		cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, ocv_result);

		int64 t1 = cv::getTickCount();

		//writerocv.write(ocv_result);

		totalTimeMs += (t1 - t0) / cv::getTickFrequency();
		// _________________________



		//___________fpga__________
		int64 t0_hard = cv::getTickCount();

	    msync(virtual_src_addr, img_size, MS_SYNC);

	    write_dma(dma_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, img_size);
	    write_dma(dma_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, img_size);

		dma_mm2s_sync(dma_virtual_addr);
		dma_s2mm_sync(dma_virtual_addr);


		memcpy(fast_buf.data(), virtual_dst_addr, img_size);

		int64 t1_hard = cv::getTickCount();
		totalTimeMs_hard += (t1_hard - t0_hard) / cv::getTickFrequency();

		frameCount++;
		printf("frame finished\n");

		PSNR += getPSNR(ocv_result, fast_mat);
		MSE += computeMSE(ocv_result, fast_mat);
		SSIM += computeSSIM(ocv_result, fast_mat);


	}

	cap.release();
	PSNR /= frameCount;
	MSE /= frameCount;
	SSIM /= frameCount;
	printf("MSE: %f\n", MSE);
	printf("PSNR: %f\n", PSNR);
	printf("SSIM: %f\n", SSIM);

	double avgMs = 1000.0 * totalTimeMs / frameCount;
	printf("Average processing time per frame: %.3f ms\n", avgMs);

	double avgMs_hard = 1000.0 * totalTimeMs_hard / frameCount;
	printf("Average processing time fpga per frame: %.3f ms\n", avgMs_hard);

	cv::imwrite("src_frame.png", resized);
	cv::imwrite("fpga_frame.png", ocv_result);
	cv::imwrite("ocv_frame.png", fast_mat);

	munmap((void*)sobel_virtual_addr, 65536);
	munmap((void*)dma_virtual_addr, 65536);
	munmap((void*)virtual_src_addr, img_size);
	munmap((void*)virtual_dst_addr, img_size);
	close(ddr_memory);

	printf("finished \n");

    return 0;
}
