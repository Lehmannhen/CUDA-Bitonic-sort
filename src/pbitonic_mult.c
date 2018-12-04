/* File:    pbitonic_mult.cu
 *
 * Purpose: Implement parallel bitonic sort using multiple kernels
 *          to avoid race conditions
 *          Note that the number of elements in the list, n, should 
 *          be a power of 2.
 *
 *          This version has extensive DEBUG output with the DEBUG
 *          flag set.
 *
 * Compile: nvcc -arch=sm_30 -o pbitonic_mult pbitonic_mult.cu
 * Usage:   ./pbitonic_mult <n> <blk_ct> <th_per_blk> [mod]
 *              n:  number of elements in list
 *         blk_ct:  number of thread blocks
 *     th_per_blk:  number of threads per block  
 *            mod:  if mod is present it is used as the modulus
 *                  with the C random function to generate the
 *                  elements of the list.  If mod is not present
 *                  the user should enter the list.
 *
 * Input:   none if mod is on the command line
 *          list of n ints if mod is not on the command line
 * Output:  elapsed wall clock time for sort and whether the
 *          list is sorted
 *
 * Notes:
 * 1.  In order to see input and output lists, PRINT_LIST 
 *     at compile time
 * 2.  Very verbose output (from the kernels) is enabled with the compiler macro
 *     DEBUG
 * 
 * Author:  Henrik Lehmann
 */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define INC 0
#define DEC 1
#define MAX_TITLE 1000
#define ALL_ONES (~0)
#define U_SZ (8 * sizeof(unsigned))
#define TH_COUNT_COEFF 4

/* ----------------------- Host prototypes -----------------------------*/
void Kernel_driver(int a[], int n, int blk_ct, int th_per_blk);
void Usage(char* prog_name);
void Get_args(int argc, char *argv[], int *n_p, int *blk_ct, int *th_per_blk, int* mod_p);
void Generate_list(int a[], int n, int mod);
void Print_list(int a[], int n, char* title);
void Read_list(int a[], int n);
int  Check_sort(int a[], int n);
void Print_unsigned(unsigned val, unsigned field_width);
unsigned Get_width(unsigned val);

/* ----------------------- Device prototypes -----------------------------*/
__device__ unsigned Insert_zero(unsigned val, unsigned j);
__device__ void Compare_swap(int a[], unsigned elt, unsigned partner,
               unsigned inc_dec);
__device__ unsigned Find_bit(unsigned bin_val);
__device__ void Get_elts(unsigned th, unsigned stage, unsigned which_bit, 
               unsigned *my_elt1_p, unsigned *my_elt2_p);



/*---------------------------- Device code --------------------------------*/
/*-------------------------------------------------------------------------
 * Function:     Sort_first (kernel)
 * Purpose:      Let each thread block compute 2 * thread per block butterflies
 *               and sort its sublist
 * In/out args:  a: an array of ints
 *
 * Note:         1. This function makes each thread block turn its part of
 *                  the array a into a bitonic sequence
 *               2. Each thread block can only compute butterflies on a sublist
 *                  of size 2 * threads per block since it is not possible 
 *                  to create a barrier over multiple thread blocks unless
 *                  the device returns to the host 
 */
__global__ void Sort_first(int a[]) {
    unsigned th = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned my_elt1, my_elt2, which_bit;

    for (unsigned bf_sz = 2; bf_sz <= 2 * blockDim.x; bf_sz <<= 1) {
        for (unsigned stage = bf_sz >> 1; stage > 0; stage >>= 1) {
            which_bit = Find_bit(stage);
            Get_elts(th, stage, which_bit, &my_elt1, &my_elt2);
            Compare_swap(a, my_elt1, my_elt2, my_elt1 & bf_sz);
            __syncthreads();
        }
    }
}






/*-------------------------------------------------------------------------
 * Function:       Sort_one_stage (kernel)
 * Purpose:        Compute one stage that corresponds to the butterfly size
 * In arg:            bf_sz: the current butterfly size in the driver (on the host)
 *               curr_stage: the current stage in the driver (on the host) 
 * In/out args:           a: an array of ints
 *
 * Note:         1. This function has to return after computing one stage
 *                  to assure synchronization among the threads and thread blocks
 *                
 */
__global__ void Sort_one_stage(int a[], unsigned bf_sz, unsigned curr_stage) {
    unsigned th = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned my_elt1, my_elt2, which_bit;

    which_bit = Find_bit(curr_stage);
    Get_elts(th, curr_stage, which_bit, &my_elt1, &my_elt2);
    Compare_swap(a, my_elt1, my_elt2, my_elt1 & bf_sz);
}






/*-------------------------------------------------------------------------
 * Function:     Sort_rem_stages (kernel)
 * Purpose:      Sort the remaining stages in a butterfly
 * In arg:            bf_sz: the current butterfly size in the driver on the host
 *               curr_stage: the current stage in the driver on the host
 * In/out args:           a: an array of ints
 *
 * Note:         This function is used when the butterflies are greater than
 *               2 * threads per block and the stages are "small" enough to make 
 *               threads in a block only compare and swap values in the array a 
 *               that are within the same thread block. Meaning no thread within
 *               a thread block modifies elements that belong to another thread block
 *               and syncthreads() can be used as a barrier. 
 *               
 */
__global__ void Sort_rem_stages(int a[], unsigned bf_sz, unsigned curr_stage) {
    unsigned th = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned my_elt1, my_elt2, which_bit;

    for (unsigned stage = curr_stage; stage > 0; stage >>= 1) {
        which_bit = Find_bit(stage);
        Get_elts(th, stage, which_bit, &my_elt1, &my_elt2);
        Compare_swap(a, my_elt1, my_elt2, my_elt1 & bf_sz);
        __syncthreads();
    }

}






/*------------------------ Host code ------------------------------------*/
int main(int argc, char* argv[]) {
    int  n, mod, th_per_blk, blk_ct;
    int *a;
    double start, finish;

    Get_args(argc, argv, &n, &blk_ct, &th_per_blk, &mod);
    cudaMallocManaged(&a, n * sizeof(int));
    
    if (mod != 0)
        Generate_list(a, n, mod);
    else
        Read_list(a, n);
    
#  ifdef PRINT_LIST
    printf("n = %d, mod = %d\n", n, mod);
    Print_list(a, n, "Before sort");
#  endif
   
    GET_TIME(start);
    Kernel_driver(a, n, blk_ct, th_per_blk);
    /* Wait for kernel to complete */
    cudaDeviceSynchronize();
    GET_TIME(finish);
    
    printf("Elapsed time to sort %d ints using bitonic sort = %e seconds\n", 
           n, finish-start);
    
#  ifdef PRINT_LIST
    Print_list(a, n, "After sort");
#  endif
    
    if (Check_sort(a, n) != 0) 
        printf("List is sorted\n");
    else
        printf("List is not sorted\n");
    
   cudaFree(a);
   return 0;
} 






/*-----------------------------------------------------------------
 * Function:       Kernel_driver
 * Purpose:        To divide the work of a bitonic sort among different
 *                 kernels to avoid race conditions
 * In args:             n: the number of elements in a
 *                 blk_ct: the number of thread blocks
 *             th_per_blk: the number of threads per block
 *
 * Out args:            a: the array of ints
 *
 * Note:           This function makes sure that there are no race conditions
 *                 in a kernel.
 *                 1. The first kernel Sort_first will sort a sublist of length 
 *                    2048 since the maximum number of threads in a block is 1024
 *                    and each thread are responsible for 2 elements in the 
 *                    list in each stage per butterfly. This kernel will compute the
 *                    first 2 * th_per_blk butterflies. For bigger butterflies the 
 *                    threads will access elements belonging to threads in another
 *                    thread block. This makes synchronization impossible within the
 *                    the same kernel (with the system we have). Instead synchronization
 *                    is done by __syncthreads() which only can synch threads within
 *                    the same block and then a different kernel is called to compute the
 *                    bigger butterflies.
 *
 *                  2. For bigger butterflies and big stages > 2 * th_per_blk the kernel
 *                     Sort_one_stage is called and it only computes one stage and return.
 *
 *                  3. When stages in a butterfly > 2 * th_per_blk is smaller than or equal to
 *                     2 * th_per_blk then the remaining stages in that butterfly is computed
 *                     by the kernel Sort_rem_stages 
 */
void Kernel_driver(int a[], int n, int blk_ct, int th_per_blk) {
    unsigned sublist_sz, curr_stage;

    /* Each thread block computes the first 2 * th_per_blk butterflies */
    Sort_first<<<blk_ct, th_per_blk>>>(a);

    /* Compute butterflies greater than 2*th_per_blk */
    for (unsigned bf_sz = TH_COUNT_COEFF * th_per_blk; bf_sz <= n; bf_sz <<= 1) {
        sublist_sz = bf_sz;
	curr_stage = bf_sz >> 1;
	while (sublist_sz > 2 * th_per_blk) {
	    Sort_one_stage<<<blk_ct, th_per_blk>>>(a, bf_sz, curr_stage);
	    sublist_sz >>= 1;
            curr_stage >>= 1;           
	}
        Sort_rem_stages<<<blk_ct, th_per_blk>>>(a, bf_sz, curr_stage);
    }
}






/*-----------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Summary of how to run program
 */
void Usage(char* prog_name) {
    fprintf(stderr, "usage:   %s <n> <blk_ct> <th_per_blk> [mod]\n", prog_name);
    fprintf(stderr, "           n :  number of elements in list\n");
    fprintf(stderr, "      blk_ct :  number of thread blocks\n");
    fprintf(stderr, "  th_per_blk :  number of threads per block\n");
    fprintf(stderr, "          mod:  if present generate list using the random\n");
    fprintf(stderr, "                number generator random and modulus mod.\n");
    fprintf(stderr, "                If absent user will enter list\n");
    exit(0);
}






/*-----------------------------------------------------------------
 * Function:     Get_args
 * Purpose:      Get and check command line arguments
 * In args:      argc, argv
 * In/Out args:  n_p, blk_ct, th_per_blk, mod_p
 */
void Get_args(int argc, char* argv[], int* n_p, int *blk_ct, int *th_per_blk, int* mod_p) {
    if (argc != 4 && argc != 5)
        Usage(argv[0]);
    
    *n_p = strtol(argv[1], NULL, 10);
    *blk_ct = strtol(argv[2], NULL, 10);
    *th_per_blk = strtol(argv[3], NULL, 10);

    if (argc == 5)
        *mod_p = strtol(argv[4], NULL, 10);
    else
        *mod_p = 0;
}






/*-----------------------------------------------------------------
 * Function:  Generate_list
 * Purpose:   Use random number generator to generate list elements
 * In args:   n
 * Out args:  a
 */
void Generate_list(int a[], int n, int mod) {
    int i;

    srandom(1);
    for (i = 0; i < n; i++)
        a[i] = random() % mod;
}






/*-----------------------------------------------------------------
 * Function:  Print_list
 * Purpose:   Print the elements in the list
 * In args:   a, n
 */
void Print_list(int a[], int n, char* title) {
    int i;

    printf("%s:  ", title);
    for (i = 0; i < n; i++)
        printf("%d ", a[i]);
    printf("\n\n");
}






/*-----------------------------------------------------------------
 * Function:  Read_list
 * Purpose:   Read elements of list from stdin
 * In args:   n
 * Out args:  a
 */
void Read_list(int a[], int n) {
    int i;

    printf("Please enter the elements of the list\n");
    for (i = 0; i < n; i++)
        scanf("%d", &a[i]);
}






/*---------------------------------------------------------------------
 * Function:  Get_width
 * Purpose:   Determine the number of bits in the binary rep of val
 *            from the least significant bit to the leftmost nonzero
 *            bit.  The number of bits in zero is zero.
 */
unsigned Get_width(unsigned val) {
    unsigned field_width = 0;

    while (val != 0) {
        val >>= 1;
        field_width++;
    }
    return field_width;
}






/*---------------------------------------------------------------------
 * Function:  Print_unsigned
 * Purpose:   Print the binary representation of an unsigned int
 */
void Print_unsigned(unsigned val, unsigned field_width) {
    unsigned curr_bit, i;
    /* +1 for null char terminating string */
    char bin_str[field_width+1];
    
    for (i = 0; i < field_width; i++)
        bin_str[i] = '0';
    bin_str[field_width] = '\0';
    
    if (val == 0) {
        printf("%s", bin_str);
        return;
    }
    
    /* val != 0 */
    curr_bit = field_width-1;
    while (val > 0) {
        if (val & 1) bin_str[curr_bit] = '1';
        val >>= 1;
        curr_bit--;
    }

    printf("%s", bin_str);
}






/*-----------------------------------------------------------------
 * Function:     Check_sort
 * Purpose:      Check to see if a list is sorted in increasing order
 * In args:      a, n
 */
int Check_sort(int a[], int n) {
    int i;

    for (i = 0; i < n-1; i++)
        if (a[i] > a[i+1]) {
            printf("a[%d] = %d > %d = a[%d]\n",
                   i, a[i], a[i+1], i+1);
            return 0;
        }
    return 1; 
}






/*----------------------- Device functions ---------------------------- */
/*---------------------------------------------------------------------
 * Function:    Insert_zero
 * Purpose:     Insert a zero in the binary representation of 
 *              val between bits j and j-1
 */
__device__ unsigned Insert_zero(unsigned val, unsigned j) {
    unsigned left_bits, right_bits, left_ones, right_ones;

    left_ones = ALL_ONES << j;  
    right_ones = ~left_ones;
    left_bits = left_ones & val;
    right_bits = right_ones & val;
    return (left_bits << 1) | right_bits;
}






/*-----------------------------------------------------------------
 * Function:    Compare_swap
 * Purpose:     Compare two elements in the list, and if out of order
 *              swap:
 *
 *                 inc_dec = INC => pair should increase
 *                 inc_dec = DEC => pair should decrease
 *             
 * In args:     elt, partner:  subscripts of elements of a
 *                 elt should always be < partner
 *              inc_dec:   whether pair should increase (0)
 *                 or decrease (1)
 * In/out arg:  a:  the list
 */
__device__ void Compare_swap(int a[], unsigned elt, unsigned partner,
      unsigned inc_dec) {
    int tmp;
    
    if (inc_dec == INC) {
        if (a[elt] > a[partner]) {
            tmp = a[elt];
            a[elt] = a[partner];
            a[partner] = tmp;
        }
    }
    else {  /* inc_dec == DEC */
        if (a[elt] < a[partner]) {
            tmp = a[elt];
            a[elt] = a[partner];
            a[partner] = tmp;
        }
    }
}






/*-----------------------------------------------------------------
 * Function:   Find_bit
 * Purpose:    Find the place of the nonzero bit in bin_val
 *
 * Note:       bin_val is a power of 2.  So it has exactly one 
 *             nonzero bit.
 */
__device__ unsigned Find_bit(unsigned bin_val) {
    unsigned bit = 0;

    while (bin_val > 1) {
        bit++;
        bin_val = bin_val >> 1;
    }
    
    return bit;
}






/*-----------------------------------------------------------------
 * Function:    Get_elts
 * Purpose:     Given a ``thread rank'' th, and which_bit should
 *              be inserted, determine the subscripts of the two
 *              elements that this thread should compare-swap
 * In args:     th, stage, which_bit
 * Out args:    my_elt1_p, my_elt2_p
 */
__device__ void Get_elts(unsigned th, unsigned stage, unsigned which_bit, 
      unsigned* my_elt1_p, unsigned* my_elt2_p) {
    *my_elt1_p = Insert_zero(th, which_bit);
    *my_elt2_p = *my_elt1_p ^ stage;
}
