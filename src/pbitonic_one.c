/* File:    pbitonic_one.cu
 *
 * Purpose: Implement a parallel bitonic sort  
 *          Note that the number of elements in the list, n, should 
 *          be a power of 2.  
 *
 *          This version has extensive DEBUG output with the DEBUG
 *          flag set.
 *
 * Compile: nvcc -arch=sm_30 -o pbitonic_one pbitonic_one.cu
 * Usage:   ./pbitonic_one <n> [mod]
 *              n:  number of elements in list
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
 * 2.  Very verbose output is enables with the compiler macro
 *     DEBUG
 * 3.  This program assumes that n <= 2048 and only uses on thread block
 */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define INC 0
#define DEC 1
#define MAX_TITLE 1000
#define ALL_ONES (~0)
#define U_SZ (8*sizeof(unsigned))

/*---------------------- Host prototypes ------------------------- */
void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* n_p, int* mod_p);
void Generate_list(int a[], int n, int mod);
void Print_list(int a[], int n, char* title);
void Read_list(int a[], int n);
int  Check_sort(int a[], int n);
void Print_unsigned(unsigned val, unsigned field_width);
unsigned Get_width(unsigned val);

/*---------------------- Device prototypes ----------------------- */
__device__ unsigned Insert_zero(unsigned val, unsigned j);
__device__ void Compare_swap(int a[], unsigned elt, unsigned partner,
               unsigned inc_dec);
__device__ unsigned Find_bit(unsigned bin_val);
__device__ void Get_elts(unsigned th, unsigned stage, unsigned which_bit, 
               unsigned* my_elt1_p, unsigned* my_elt2_p);



/*---------------------- Device code------------------------------ */
/*----------------------------------------------------------------
 * Function:      Bitonic_sort
 * Purpose:       Sort at most 2048 elements with n / 2 threads using 
 *                bitonic sequences
 *
 * In args:        n: number of elements
 * In/out args:    a: in: array of ints, out: sorted array of ints
 *
 *
 */ 
__global__ void Bitonic_sort(int a[], int n) {
    unsigned th = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned my_elt1, my_elt2, which_bit;

    for (unsigned bf_sz = 2; bf_sz <= n; bf_sz <<= 1) {
        for (unsigned stage = bf_sz >> 1; stage > 0; stage >>= 1) {
            which_bit = Find_bit(stage);
            Get_elts(th, stage, which_bit, &my_elt1, &my_elt2);
            Compare_swap(a, my_elt1, my_elt2, my_elt1 & bf_sz);
            __syncthreads();
        }
    }
}


/*--------------------- Host code ------------------------------------*/
int main(int argc, char* argv[]) {
   int  n, mod, th_per_blk, blk_ct = 1;
   int *a;
   double start, finish;

   Get_args(argc, argv, &n, &mod);
   cudaMallocManaged(&a, n * sizeof(int));
   if (mod != 0)
      Generate_list(a, n, mod);
   else
      Read_list(a, n);

# ifdef PRINT_LIST
   printf("n = %d, mod = %d\n", n, mod);
   Print_list(a, n, "Before sort");
# endif

   th_per_blk = n / 2;
   GET_TIME(start);
   Bitonic_sort<<<blk_ct, th_per_blk>>>(a, n);
   /* Wait for kernel to complete */
   cudaDeviceSynchronize();
   GET_TIME(finish);
   
   printf("Elapsed time to sort %d ints using bitonic sort = %e seconds\n", 
         n, finish-start);

# ifdef PRINT_LIST
   Print_list(a, n, "After sort");
# endif

   if (Check_sort(a, n) != 0) 
      printf("List is sorted\n");
   else
      printf("List is not sorted\n");
   
   cudaFree(a);
   return 0;
} 






/*-----------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Summary of how to run program
 */
void Usage(char* prog_name) {
   fprintf(stderr, "usage:   %s <n> [mod]\n", prog_name);
   fprintf(stderr, "   n :  number of elements in list\n");
   fprintf(stderr, "  mod:  if present generate list using the random\n");
   fprintf(stderr, "        number generator random and modulus mod.\n");
   fprintf(stderr, "        If absent user will enter list\n");
   exit(0);
}






/*-----------------------------------------------------------------
 * Function:  Get_args
 * Purpose:   Get and check command line arguments
 * In args:   argc, argv
 * Out args:  n_p, mod_p
 */
void Get_args(int argc, char* argv[], int* n_p, int* mod_p) {
    if (argc != 2 && argc != 3)
        Usage(argv[0]);
    
    *n_p = strtol(argv[1], NULL, 10);

    if (argc == 3)
        *mod_p = strtol(argv[2], NULL, 10);
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






/* ------------------------ Device functions ---------------------- */
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
