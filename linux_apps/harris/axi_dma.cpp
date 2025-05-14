#include "axi_dma.h"

unsigned int write_dma(unsigned int *virtual_addr, int offset, unsigned int value)
{
    virtual_addr[offset >> 2] = value;
    return 0;
}

unsigned int read_dma(unsigned int *virtual_addr, int offset)
{
    return virtual_addr[offset >> 2];
}

void dma_s2mm_status(unsigned int *virtual_addr)
{
    unsigned int status = read_dma(virtual_addr, S2MM_STATUS_REGISTER);
    printf("Stream to memory-mapped status (0x%08x@0x%02x):", status, S2MM_STATUS_REGISTER);

    if (status & STATUS_HALTED) printf(" Halted.\n");
    else printf(" Running.\n");

    if (status & STATUS_IDLE) printf(" Idle.\n");
    if (status & STATUS_SG_INCLDED) printf(" SG is included.\n");
    if (status & STATUS_DMA_INTERNAL_ERR) printf(" DMA internal error.\n");
    if (status & STATUS_DMA_SLAVE_ERR) printf(" DMA slave error.\n");
    if (status & STATUS_DMA_DECODE_ERR) printf(" DMA decode error.\n");
    if (status & STATUS_SG_INTERNAL_ERR) printf(" SG internal error.\n");
    if (status & STATUS_SG_SLAVE_ERR) printf(" SG slave error.\n");
    if (status & STATUS_SG_DECODE_ERR) printf(" SG decode error.\n");
    if (status & STATUS_IOC_IRQ) printf(" IOC interrupt occurred.\n");
    if (status & STATUS_DELAY_IRQ) printf(" Interrupt on delay occurred.\n");
    if (status & STATUS_ERR_IRQ) printf(" Error interrupt occurred.\n");
}

void dma_mm2s_status(unsigned int *virtual_addr)
{
    unsigned int status = read_dma(virtual_addr, MM2S_STATUS_REGISTER);
    printf("Memory-mapped to stream status (0x%08x@0x%02x):", status, MM2S_STATUS_REGISTER);

    if (status & STATUS_HALTED) printf(" Halted.\n");
    else printf(" Running.\n");

    if (status & STATUS_IDLE) printf(" Idle.\n");
    if (status & STATUS_SG_INCLDED) printf(" SG is included.\n");
    if (status & STATUS_DMA_INTERNAL_ERR) printf(" DMA internal error.\n");
    if (status & STATUS_DMA_SLAVE_ERR) printf(" DMA slave error.\n");
    if (status & STATUS_DMA_DECODE_ERR) printf(" DMA decode error.\n");
    if (status & STATUS_SG_INTERNAL_ERR) printf(" SG internal error.\n");
    if (status & STATUS_SG_SLAVE_ERR) printf(" SG slave error.\n");
    if (status & STATUS_SG_DECODE_ERR) printf(" SG decode error.\n");
    if (status & STATUS_IOC_IRQ) printf(" IOC interrupt occurred.\n");
    if (status & STATUS_DELAY_IRQ) printf(" Interrupt on delay occurred.\n");
    if (status & STATUS_ERR_IRQ) printf(" Error interrupt occurred.\n");
}

int dma_mm2s_sync(unsigned int *virtual_addr)
{
    unsigned int mm2s_status;

    do {
        mm2s_status = read_dma(virtual_addr, MM2S_STATUS_REGISTER);
    } while (!(mm2s_status & IOC_IRQ_FLAG));

    if (mm2s_status & (STATUS_DMA_INTERNAL_ERR | STATUS_DMA_SLAVE_ERR | STATUS_DMA_DECODE_ERR)) {
        printf("MM2S DMA Error: 0x%08X\n", mm2s_status);
        return -1;
    }

    return 0;
}

int dma_s2mm_sync(unsigned int *virtual_addr)
{
    unsigned int s2mm_status;

    do {
        s2mm_status = read_dma(virtual_addr, S2MM_STATUS_REGISTER);
    } while (!(s2mm_status & IOC_IRQ_FLAG));

    if (s2mm_status & (STATUS_DMA_INTERNAL_ERR | STATUS_DMA_SLAVE_ERR | STATUS_DMA_DECODE_ERR)) {
        printf("S2MM DMA Error: 0x%08X\n", s2mm_status);
        return -1;
    }

    return 0;
}
