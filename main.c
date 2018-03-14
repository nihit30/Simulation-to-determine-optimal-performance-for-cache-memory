#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
a[k][l]=h+s*(g-h*tau);

/*Macros*/
#define MAX_LINES        16384
#define MAX_WAYS        16

#define WR_BACK            0
#define WR_ALLOC        1
#define WR_NON_ALLOC    2

#define SET                1
#define CLEAR            0

int nrot;          // no. of rotations required for jacobi for convergence (3n^2 to 5n^2)   n = size of subscript of matrix
int n = 128;
double a[128][128]__attribute__((aligned(4)));
int bl=0;
int set_size=0;
/*Cache Arrays*/
bool valid_bit[MAX_LINES][MAX_WAYS];
bool dirty_bit[MAX_LINES][MAX_WAYS];
uint32_t tag[MAX_LINES][MAX_WAYS];
uint8_t lru[MAX_LINES][MAX_WAYS];

/*Cache Counters*/
uint32_t cntr_wr_mem, cntr_wr_line, cntr_wr_miss, cntr_wr_d_replace, cntr_wr_cache, cntr_wr_through_mem;
uint32_t cntr_rd_mem, cntr_rd_line, cntr_rd_miss, cntr_rd_d_replace, cntr_rd_cache,cntr_dirty;
uint32_t cntr_wr_mem_fun, cntr_rd_mem_fun;

/*Globals*/
uint32_t CURR_WAY, CURR_WR_STRATEGY, CURR_B_LEN, CURR_MASK_TAG;
uint8_t line_bits_cnt, block_offset_bits_cnt, tag_bits_cnt;
uint32_t ReadBlockTimer, wr_through_data_tx_cntr;
uint16_t CURR_LINES_CNT, CURR_MASK_LINE;

/*Function Declaratios*/
void init_cache(void);
void writeMemory(void *, uint8_t);
void writeLine(void*);
void writeCache(uint16_t, uint8_t);
void readMemory(void *, uint8_t);
void readLine(void*);
bool isMiss(uint32_t);
uint16_t GetLine(uint32_t);
uint32_t GetTag(uint32_t);
uint8_t GetLRUWay(uint16_t);
bool isDirty(uint16_t, uint8_t);
void writeBlock(void*);
void clearDirty(uint16_t, uint8_t);
void setDirty(uint32_t);
void clearTag(uint16_t, uint8_t);
void setTag(uint16_t, uint8_t, uint32_t);
void inValidate(uint16_t, uint8_t);
void readBlock(void*);
void Validate(uint16_t, uint8_t);
void UpdateLRU(uint32_t);
void readCache(uint16_t, uint8_t);

uint8_t get_Offset_Bits_Count(uint8_t);
uint16_t get_Number_of_Lines(uint8_t, uint8_t);
uint8_t get_Line_Bits_Count(uint16_t);

void test_cases(void);


/* CAUTION: This is the ANSI C (only) version of the Numerical Recipes
 utility file nrutil.c.  Do not confuse this file with the same-named
 file nrutil.c that is supplied in the same subdirectory or archive
 as the header file nrutil.h.  *That* file contains both ANSI and
 traditional K&R versions, along with #ifdef macros to select the
 correct version.  *This* file contains only ANSI C.               */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr,"%s\n",error_text);
    fprintf(stderr,"...now exiting to system...\n");
    exit(1);
}

double *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
    double *v;
    
    v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
    if (!v) nrerror("allocation failure in vector()");
    return v-nl+NR_END;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
    int *v;
    
    v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
    if (!v) nrerror("allocation failure in ivector()");
    return v-nl+NR_END;
}

unsigned char *cvector(long nl, long nh)
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
    unsigned char *v;
    
    v=(unsigned char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
    if (!v) nrerror("allocation failure in cvector()");
    return v-nl+NR_END;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
    unsigned long *v;
    
    v=(unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
    if (!v) nrerror("allocation failure in lvector()");
    return v-nl+NR_END;
}

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
    double *v;
    
    v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
    if (!v) nrerror("allocation failure in dvector()");
    return v-nl+NR_END;
}

double **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    double **m;
    
    /* allocate pointers to rows */
    m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
    if (!m) nrerror("allocation failure 1 in matrix()");
    m += NR_END;
    m -= nrl;
    
    /* allocate rows and set pointers to them */
    m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
    if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
    m[nrl] += NR_END;
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    /* return pointer to array of pointers to rows */
    return m;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    double **m;
    
    /* allocate pointers to rows */
    m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
    if (!m) nrerror("allocation failure 1 in matrix()");
    m += NR_END;
    m -= nrl;
    
    /* allocate rows and set pointers to them */
    m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
    if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
    m[nrl] += NR_END;
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    /* return pointer to array of pointers to rows */
    return m;
}

int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    int **m;
    
    /* allocate pointers to rows */
    m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
    if (!m) nrerror("allocation failure 1 in matrix()");
    m += NR_END;
    m -= nrl;
    
    
    /* allocate rows and set pointers to them */
    m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
    if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
    m[nrl] += NR_END;
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    /* return pointer to array of pointers to rows */
    return m;
}

float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
                  long newrl, long newcl)
/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
    long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
    float **m;
    
    /* allocate array of pointers to rows */
    m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
    if (!m) nrerror("allocation failure in submatrix()");
    m += NR_END;
    m -= newrl;
    
    /* set pointers to rows */
    for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;
    
    /* return pointer to array of pointers to rows */
    return m;
}

double **convert_matrix(double *a, long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
 declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
 and ncol=nch-ncl+1. The routine should be called with the address
 &a[0][0] as the first argument. */
{
    long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
    double **m;
    
    /* allocate pointers to rows */
    m=(double **) malloc((size_t) ((nrow+NR_END)*sizeof(double*)));
    if (!m) nrerror("allocation failure in convert_matrix()");
    m += NR_END;
    m -= nrl;
    
    /* set pointers to rows */
    m[nrl]=a-ncl;
    for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
    /* return pointer to array of pointers to rows */
    return m;
}

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
    long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
    float ***t;
    
    /* allocate pointers to pointers to rows */
    t=(float ***) malloc((size_t)((nrow+NR_END)*sizeof(float**)));
    if (!t) nrerror("allocation failure 1 in f3tensor()");
    t += NR_END;
    t -= nrl;
    
    /* allocate pointers to rows and set pointers to them */
    t[nrl]=(float **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float*)));
    if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
    t[nrl] += NR_END;
    t[nrl] -= ncl;
    
    /* allocate rows and set pointers to them */
    t[nrl][ncl]=(float *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(float)));
    if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
    t[nrl][ncl] += NR_END;
    t[nrl][ncl] -= ndl;
    
    for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
    for(i=nrl+1;i<=nrh;i++) {
        t[i]=t[i-1]+ncol;
        t[i][ncl]=t[i-1][ncl]+ncol*ndep;
        for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
    }
    
    /* return pointer to array of pointers to rows */
    return t;
}

void free_vector(double *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void free_cvector(unsigned char *v, long nl, long nh)
/* free an unsigned char vector allocated with cvector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
    free((FREE_ARG) (m[nrl]+ncl-NR_END));
    free((FREE_ARG) (m+nrl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
    free((FREE_ARG) (m[nrl]+ncl-NR_END));
    free((FREE_ARG) (m+nrl-NR_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated by imatrix() */
{
    free((FREE_ARG) (m[nrl]+ncl-NR_END));
    free((FREE_ARG) (m+nrl-NR_END));
}

void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
    free((FREE_ARG) (b+nrl-NR_END));
}

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a matrix allocated by convert_matrix() */
{
    free((FREE_ARG) (b+nrl-NR_END));
}

void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
                   long ndl, long ndh)
/* free a float f3tensor allocated by f3tensor() */
{
    free((FREE_ARG) (t[nrl][ncl]+ndl-NR_END));
    free((FREE_ARG) (t[nrl]+ncl-NR_END));
    free((FREE_ARG) (t+nrl-NR_END));
}


/*
 This Function emulates writting to main memory from cache.
 */
void writeBlock(void * addr)
{
    cntr_wr_mem += 1;
}


/*
 This Function emulates writting to cache from processor.
 */
void writeCache(uint16_t line, uint8_t way)
{
    cntr_wr_cache += 1;
}

/*
 This function initialises cache; by clearing all cache metadata.
 This function would be called on every cpu &/ cache reset.
 */
void init_cache(void)
{
    uint16_t line = 0;
    uint8_t way = 0;
    
    /*clearing all cache metadata*/
    for (line = 0; line < MAX_LINES; line++)
    {
        for (way = 0; way < MAX_WAYS; way++)
        {
            valid_bit[line][way] = CLEAR;
            dirty_bit[line][way] = CLEAR;
            tag[line][way] = CLEAR;
            lru[line][way] = CLEAR;
        }
    }
    
    /*clearing counters at the beginning*/
    /*Write Counters*/
    cntr_wr_mem = CLEAR;
    cntr_wr_line = CLEAR;
    cntr_wr_miss = CLEAR;
    cntr_wr_d_replace = CLEAR;
    cntr_wr_cache = CLEAR;
    cntr_wr_through_mem = CLEAR;
    /*Read Counters*/
    cntr_rd_mem = CLEAR;
    cntr_rd_line = CLEAR;
    cntr_rd_miss = CLEAR;
    cntr_rd_d_replace = CLEAR;
    cntr_rd_cache = 0;
}

/*
 This is the main write function.
 In this, issue of data size > block/way size is taken care of.
 */
void writeMemory(void* mem_addr, uint8_t size)
{
    int old_line = -1;
    uint8_t ctr = 0;
    uint32_t addr = 0;
    uint32_t line = 0;
    
    if ((size) <= (CURR_B_LEN * 4))
    {
        if(size <= 4)
            wr_through_data_tx_cntr = 1;
        else
            wr_through_data_tx_cntr = 2;
    }
    else if ((size) > (CURR_B_LEN * 4))
    {
        wr_through_data_tx_cntr = 1;
    }
    
    
    for (ctr = 0; ctr < size; ctr++)
    {
        addr = (uint32_t)mem_addr + ctr;
        line = GetLine(addr);
        
        /*
         This avoids false line access. Thus avoiding false writes.
         */
        if (line != old_line)
        {
            writeLine((void*)addr);
            old_line = line;
        }
    }
    
    cntr_wr_mem_fun += 1;
}



/*
 This function is used to write a whole block to a line.
 
 HIT                               MISS
 
 CACHE            MEM            CACHE            MEM
 
 WB              Y                 N              Y                 N
 
 WTA              Y                 Y              Y                 Y
 
 WTNA          Y                 Y              N                 Y
 
 */
void writeLine(void* mem_addr)
{
    uint32_t addr = (uint32_t)mem_addr;
    uint16_t line = GetLine(addr);
    uint8_t  way = 1,cnt = 0;
    if ((line == 0x1054) /*&& (way == 1)*/)
    {
        way--;
        way += 1;
    }
    /*uint8_t*/ way = GetLRUWay(line);
    
    
    /*
     If its a write miss, we need to update cache in wb & wta only.
     During process we need to check for dirty & update mem accordingly.
     */
    if ((CURR_WR_STRATEGY == WR_BACK) || (CURR_WR_STRATEGY == WR_ALLOC))
    {
        if (isMiss(addr))
        {
            line = GetLine(addr);
            way = GetLRUWay(line);
            
            if (isDirty(line, way))
            {
                
                writeBlock((void *) addr);
                clearDirty(line, way);
                
                cntr_wr_d_replace += 1;
            }
            
            clearTag(line, way);
            inValidate(line, way);
            readBlock((void *)addr);
            setTag(line, way, addr);
            Validate(line, way);
            
            cntr_wr_miss += 1;
        }
        
        writeCache(line, way);
        if (CURR_WR_STRATEGY == WR_BACK)
        {
            setDirty(addr);
            cntr_dirty += 1;
        }
        UpdateLRU(addr);
    }
    
    if (CURR_WR_STRATEGY == WR_NON_ALLOC)
    {
        if (!isMiss(addr))
        {
            writeCache(line, way);
            UpdateLRU(addr);
        }
        else if (isMiss(addr))
        {
            cntr_wr_miss += 1;
        }
    }
    
    if ((CURR_WR_STRATEGY == WR_ALLOC) || (CURR_WR_STRATEGY == WR_NON_ALLOC))   //BL = 1;
    {
        for (cnt = 0; cnt < wr_through_data_tx_cntr; cnt++)
        {
            writeBlock((void *)addr);
            cntr_wr_through_mem += 1;
        }
    }
    
    cntr_wr_line += 1;
}


/*
 This is the main read function.
 In this, issue of data size > block/way size is taken care of.
 */
void readMemory(void * addr, uint8_t size)
{
    uint8_t ctr = 0;
    uint32_t temp_addr = 0, temp_line = 0;
    int32_t old_line = -1;  //-1 is put just to take care of very first time if the line bits are zero.
    
    //printf(" %x %d\n", (uint32_t)addr, size);
    
    for (ctr = 0; ctr < size; ctr++)
    {
        temp_addr = (uint32_t)addr + ctr;
        //printf(" %x\n", (uint32_t)temp_addr);
        
        temp_line = GetLine(temp_addr);
        
        /*
         Suppose if we are reading 8 byte data & way size is 4.
         It will read 4 bytes in 1 read & store 2 diff '4 bytes' in 2 diff lines.
         As we are increamenting ctr or addr by 1 in this for loop; index bits will be same for 4 times.
         So to avoid extra 3 reads, we are using below if case.
         */
        if (temp_line != old_line)
        {
            readLine((void *)temp_addr);
            old_line = temp_line;
        }
        
        //old_line = temp_line;
    }
    
    cntr_rd_mem_fun += 1;
}


/*
 This function is used to read a whole block in a line. We always read from cache.
 So if its a miss, we read from main mem. and copy that to cache & read from cache.
 Before reading from main mem, in case of write back, wecheck for dirty bit & write back in main mem.
 */
void readLine(void * addr)
{
    uint32_t mem_addr = (uint32_t)addr;
    uint16_t line = 0;
    uint8_t way = 0;
    
    if (isMiss(mem_addr))
    {
        line = GetLine(mem_addr);
        way = GetLRUWay(line);
        
        if (isDirty(line, way))
        {
            writeBlock((void *) mem_addr);
            clearDirty(line, way);
            
            cntr_rd_d_replace += 1;
        }
        
        clearTag(line, way);
        inValidate(line, way);
        readBlock((void *)mem_addr);
        setTag(line, way, mem_addr);
        Validate(line, way);
        
        cntr_rd_miss += 1;
    }
    
    UpdateLRU(mem_addr);
    readCache(line, way);
    
    cntr_rd_line += 1;
}

/*
 This Function is used to check cache miss/hit.
 This returns bool false on cache hit & bool true on cache miss.
 This functions first separates out line & tag bits.
 If tag bits match in particular line, then its a hit.
 This logic holds true only if valid bit of that way is set.
 */
bool isMiss(uint32_t addr)
{
    uint32_t temp_tag = 0;
    uint16_t line = 0;
    uint8_t cntr = 0;
    
    line = GetLine(addr);
    temp_tag = GetTag(addr);
    
    for (cntr = 0; cntr < CURR_WAY; cntr++)
    {
        if (valid_bit[line][cntr] == SET)
        {
            if (tag[line][cntr] == temp_tag)
            {
                return false;        // this is cache hit
            }
        }
    }
    
    return true;   // if theres no match of tag bits, we will return miss i.e. true
}

/*
 This Function is used calculate line/index bits.
 CURR_MASK_LINE is used to mask out tag bits. It is set in main fun.
 */
inline uint16_t GetLine(uint32_t addr)
{
    uint32_t line_bits = 0;
    line_bits = /*(uint16_t)*/(addr >> block_offset_bits_cnt);
    line_bits = line_bits & CURR_MASK_LINE;
    return line_bits;
}

/*
 This Function is used calculate tag bits.
 CURR_MASK_TAG is used to mask out ramaining bits in addr. It is set in main fun.
 */
inline uint32_t GetTag(uint32_t addr)
{
    uint32_t tag_bits = 0;
    tag_bits = addr >> (block_offset_bits_cnt + line_bits_cnt);
    //tag_bits = tag_bits & CURR_MASK_TAG;
    return tag_bits;
}

/*
 This Function is used to get way to copy data in cache.
 We will 1st check for empty/invalid  block in the given line.
 If line is full/valid , we will go for lru way.
 */
uint8_t GetLRUWay(uint16_t line)
{
    uint8_t ctr = 0, max_lru = lru[line][0],max_lru_way = 0;
    
    for (ctr = 0; ctr < CURR_WAY; ctr++)
    {
        if (valid_bit[line][ctr] == CLEAR)
        {
            return ctr;
        }
    }
    
    for (ctr = 1; ctr < CURR_WAY; ctr++)
    {
        if (valid_bit[line][ctr] == SET)
        {
            if (max_lru < lru[line][ctr])
            {
                max_lru = lru[line][ctr];
                max_lru_way = ctr;
            }
        }
    }
    
    return max_lru_way;
}


/*
 This Function checks weather given way in the line is Dirty or not.
 */
inline bool isDirty(uint16_t line, uint8_t way)
{
    bool ret_val = false;
    
    (dirty_bit[line][way] == SET) ? (ret_val = true) : (ret_val = false);
    
    return ret_val;
}


/*
 This Function clears dirty bit corresponding to given way in the given line.
 */
inline void clearDirty(uint16_t line, uint8_t way)
{
    dirty_bit[line][way] = CLEAR;
}


/*
 This Function clears dirty bit corresponding to given way in the given line.
 */
inline void setDirty(uint32_t addr)
{
    uint32_t temp_tag = 0;
    uint16_t line = 0;
    uint8_t cntr = 0;
    
    line = GetLine(addr);
    temp_tag = GetTag(addr);
    
    for (cntr = 0; cntr < CURR_WAY; cntr++)
    {
        if (valid_bit[line][cntr] == SET)
        {
            if (tag[line][cntr] == temp_tag)
            {
                dirty_bit[line][cntr] = SET;
            }
        }
    }
}


/*
 This Function clears tag bits corresponding to given way in the given line.
 */
inline void clearTag(uint16_t line, uint8_t way)
{
    tag[line][way] = CLEAR;
}


/*
 This Function set tag bits corresponding to given way in the given line.
 */
inline void setTag(uint16_t line, uint8_t way, uint32_t addr)
{
    uint32_t temp_tag = 0;
    
    temp_tag = GetTag(addr);
    
    tag[line][way] = temp_tag;
}

/*
 This Function clears valid bit corresponding to given way in the given line.
 */
inline void inValidate(uint16_t line, uint8_t way)
{
    valid_bit[line][way] = CLEAR;
}

/*
 This Function emulates reading from main memory to cache.
 */
void readBlock(void * addr)
{
    cntr_rd_mem += 1;//ReadBlockTimer += (90 + 15 * (CURR_B_LEN - 1)); //reading from main memory
    //ReadBlockTimer += 1;  // writting into cache.  DOUBT
}

/*
 This Function sets valid bit corresponding to given way in the given line.
 */
inline void Validate(uint16_t line, uint8_t way)
{
    valid_bit[line][way] = SET;
}

/*
 This function updates LRU.
 Decreament incoming lru if greater than 0.
 Age only lower or equal values, dont touch higher values.
 */
void UpdateLRU(uint32_t addr)
{
    uint32_t temp_tag = 0;
    uint16_t line = 0;
    uint8_t way = 0, ctr = 0;
    
    line = GetLine(addr);
    temp_tag = GetTag(addr);
    //way = GetWay(addr);
    for (ctr = 0; ctr < CURR_WAY; ctr++)
    {
        if (valid_bit[line][ctr] == SET)
        {
            if (tag[line][ctr] == temp_tag)
            {
                way = ctr;
                break;
            }
        }
    }
    
    for (ctr = 0; ctr < CURR_WAY; ctr++)
    {
        if ((ctr != way) && (lru[line][ctr] <= lru[line][way]) && (valid_bit[line][ctr] == SET))
        {
            lru[line][ctr] += 1;
        }
    }
    
    lru[line][way] = 0;
    
}

/*
 This Function emulates reading from cache to processor.
 */
void readCache(uint16_t line, uint8_t way)
{
    cntr_rd_cache += 1;
}

uint8_t get_Offset_Bits_Count(uint8_t b_len)
{
    uint16_t block_size = 4 * b_len;
    
    uint8_t count = (log(block_size) / log(2));
    
    return count;
}

inline uint16_t get_Number_of_Lines(uint8_t way, uint8_t b_len)
{
    uint16_t line_cnt = MAX_LINES;
    
    if (way == 0)
    {
        way = 1;
    }
    if (b_len == 0)
    {
        b_len = 1;
    }
    
    line_cnt /= way;
    line_cnt /= b_len;
    
    return line_cnt;
}

uint8_t get_Line_Bits_Count(uint16_t total_lines)
{
    if (total_lines == 128)
    {
        CURR_MASK_LINE = 0b1111111;
        return 7;
    }
    else if (total_lines == 256)
    {
        CURR_MASK_LINE = 0b11111111;
        return 8;
    }
    else if (total_lines == 512)
    {
        CURR_MASK_LINE = 0b111111111;
        return 9;
    }
    else if (total_lines == 1024)
    {
        CURR_MASK_LINE = 0b1111111111;
        return 10;
    }
    else if (total_lines == 2048)
    {
        CURR_MASK_LINE = 0b11111111111;
        return 11;
    }
    else if (total_lines == 4096)
    {
        CURR_MASK_LINE = 0b111111111111;
        return 12;
    }
    else if (total_lines == 8192)
    {
        CURR_MASK_LINE = 0b1111111111111;
        return 13;
    }
    else if (total_lines == 16384)
    {
        CURR_MASK_LINE = 0b11111111111111;
        return 14;
    }
    else
    {
        CURR_MASK_LINE = 0b11111111111111;
        return 14;
    }
}


void test_cases(void)
{
    uint8_t i;
    
    printf("%u %u %u\n", CURR_WR_STRATEGY, CURR_WAY, CURR_B_LEN);
    
    printf("block_offset_bits_cnt : %d \n", block_offset_bits_cnt);
    printf("CURR_LINES_CNT : %d \n", CURR_LINES_CNT);
    printf("line_bits_cnt : %d \n", line_bits_cnt);
    printf("tag_bits_cnt : %d \n", tag_bits_cnt);
    printf("CURR_MASK_LINE : %x \n", CURR_MASK_LINE);
    
    printf("LINE : %x \n", GetLine(0xAABCCDE1));
    printf(" TAG : %x \n", GetTag(0xAABCCDE1));
    
    printf("LINE : %x \n", GetLRUWay(0));
    
    valid_bit[0][0] = valid_bit[0][1] = SET;
    printf("LINE : %x \n", GetLRUWay(0));
    
    valid_bit[0][0] = valid_bit[0][1] = SET;
    valid_bit[0][2] = valid_bit[0][3] = SET;
    lru[0][0] = lru[0][1] = lru[0][2] = 2;
    lru[0][3] = 3;
    printf("LINE : %x \n", GetLRUWay(0));
    
    valid_bit[0][0] = valid_bit[0][1] = SET;
    valid_bit[0][2] = valid_bit[0][3] = SET;
    valid_bit[0][3] = CLEAR;
    lru[0][0] = 0;
    lru[0][1] = 2;
    lru[0][2] = 3;
    lru[0][3] = 0;
    tag[0][1] = 0x2AAF3;
    UpdateLRU(0xAABCC001);
    for (i = 0; i < CURR_WAY; i++)
    {
        printf("%d \n", lru[0][i]);
    }
}


void jacobi(double **a, int n, double d[], double **v, int *nrot)
//Computes all eigenvalues and eigenvectors of a real symmetric matrix a[1..n][1..n]. On
//output, elements of a above the diagonal are destroyed. d[1..n] returns the eigenvalues of a.
//v[1..n][1..n] is a matrix whose columns contain, on output, the normalized eigenvectors of
//a. nrot returns the number of Jacobi rotations that were required.
{
    /*we will push function parameters into stack/memory*/
    writeMemory(&a, sizeof(a));
    writeMemory(&n, sizeof(n));
    writeMemory(&d, sizeof(d));
    writeMemory(&v, sizeof(v));
    writeMemory(&nrot, sizeof(nrot));
    
    //__declspec(align(32)) int j, iq, ip, i;
    int j, iq, ip, i;
    double tresh, theta, tau, t, sm, s, h, g, c, *b, *z;
    /*allocating memory for variables*/
    writeMemory(&j, sizeof(j));
    writeMemory(&iq, sizeof(iq));
    writeMemory(&ip, sizeof(ip));
    writeMemory(&i, sizeof(i));
    
    //__declspec(align(32)) double tresh, theta, tau, t, sm, s, h, g, c, *b, *z;
    /*allocating memory for variables*/
    writeMemory(&tresh, sizeof(tresh));
    writeMemory(&theta, sizeof(theta));
    writeMemory(&tau, sizeof(tau));
    writeMemory(&t, sizeof(t));
    writeMemory(&sm, sizeof(sm));
    writeMemory(&s, sizeof(s));
    writeMemory(&h, sizeof(h));
    writeMemory(&g, sizeof(g));
    writeMemory(&c, sizeof(c));
    writeMemory(&b, sizeof(b));
    writeMemory(&z, sizeof(z));
    
    
    
    readMemory(&n, sizeof(n));    /*Read n to pass to dvector fun*/
    b = dvector(1, n);
    writeMemory(&b, sizeof(b));      /*write updated b*/
    
    readMemory(&n, sizeof(n));
    z = dvector(1, n);
    writeMemory(&z, sizeof(z));
    
    writeMemory(&ip, sizeof(ip));   //write ip = 1;
    readMemory(&ip, sizeof(ip));    // Read ip & n for comparing. We will keep n in cpu register.
    readMemory(&n, sizeof(n));
    for (ip = 1; ip <= n; ip++)
    {
        writeMemory(&iq, sizeof(iq));  //write iq = 1;
        readMemory(&iq, sizeof(iq));   // Read iq for comparing. We will keep iq in cpu register for whole loop.
        readMemory(&ip, sizeof(ip));   //Read ip for below loop. Will keep ip in cpu reg.
        for (iq = 1; iq <= n; iq++)
        {
            v[ip][iq] = 0.0;
            writeMemory(&v[ip][iq], sizeof(v[ip][iq]));  //write v[ip][iq]
        }
        writeMemory(&iq, sizeof(iq));  // write iq back to mem from cpu register.
        
        // ip  is already in cpu register.
        v[ip][ip] = 1.0;
        writeMemory(&v[ip][ip], sizeof(v[ip][ip])); //write v[ip][iq]
        
        // for ip & n required for comparison, they are already in cpu register.
        writeMemory(&ip, sizeof(ip));  // Write Updated ip(i.e. ip++)
    }
    
    writeMemory(&ip, sizeof(ip));  //write ip = 1;
    readMemory(&ip, sizeof(ip));   // Read ip & n for comparing. We will keep n in cpu register.
    readMemory(&n, sizeof(n));
    for (ip = 1; ip <= n; ip++)
    {
        readMemory(&ip, sizeof(ip));     // Read ip for comparing. We will keep ip in cpu register for 1 iteration of loop.
        
        readMemory(&a[ip][ip], sizeof(a[ip][ip]));  // Read a[ip][ip].
        b[ip] = d[ip] = a[ip][ip];
        writeMemory(&d[ip], sizeof(d[ip]));  // Write updated d[ip][ip].
        writeMemory(&b[ip], sizeof(b[ip]));  // Write updated b[ip][ip].
        
        z[ip] = 0.0;
        writeMemory(&z[ip], sizeof(z[ip]));  // Write z[ip][ip].
        
        // ip & n required for comparison, they are already in cpu register.
        writeMemory(&ip, sizeof(ip));  // Write Updated ip(i.e. ip++)
    }
    
    readMemory(&nrot, sizeof(nrot));  // Read *nrot
    *nrot = 0;
    writeMemory(&nrot, sizeof(nrot));  // Write *nrot
    
    writeMemory(&i, sizeof(i));  // Write i=1;
    readMemory(&i, sizeof(i));   // Read i for comparison.
    for (i = 1; i <= 50; i++)
    {
        
        sm = 0.0;
        writeMemory(&sm, sizeof(sm));  // Write sm
        
        
        writeMemory(&ip, sizeof(ip));   // Write ip = 1;
        readMemory(&ip, sizeof(ip));    //Read ip & n for comparison. We will keep n in cpu reg. for whole for loop.
        readMemory(&n, sizeof(n));
        for (ip = 1; ip <= n - 1; ip++)
        {
            
            readMemory(&ip, sizeof(ip));  //read ip & keep it in cpu reg for below loop.
            
            writeMemory(&iq, sizeof(iq)); //write iq = ip + 1
            readMemory(&iq, sizeof(iq));  // read iq & n for comparison. We will keep n in cpu reg.
            readMemory(&n, sizeof(n));
            for (iq = ip + 1; iq <= n; iq++)
            {
                // ip is in cpu reg
                readMemory(&iq, sizeof(iq));  //read iq & keep it in cpu reg.
                readMemory(&a[ip][iq], sizeof(a[ip][iq])); // read a & sm
                readMemory(&sm, sizeof(sm));
                sm = sm + fabs(a[ip][iq]);
                writeMemory(&sm, sizeof(sm));  // write sm
                
                // for iq & n required for comparison, they are already in cpu register.
                writeMemory(&iq, sizeof(iq)); // write iq++
            }
            
            readMemory(&ip, sizeof(ip));       // for loop profiling
            // ip & n required for comparison, n is already in cpu register.
            writeMemory(&ip, sizeof(ip));   // write ip++
        }
        
        readMemory(&sm, sizeof(sm));
        if (sm == 0.0)
        {
            readMemory(&n, sizeof(n)); // read n & z. Keep n in cpu reg.
            readMemory(&z, sizeof(z));
            free_dvector(z, 1, n);
            
            readMemory(&b, sizeof(b)); // read b. n is in cpu reg.
            free_dvector(b, 1, n);
            return;
        }
        
        readMemory(&i, sizeof(i));
        if (i<4)
        {
            readMemory(&n, sizeof(n));
            readMemory(&sm, sizeof(sm));
            tresh = 0.2*sm / (n*n);
            writeMemory(&tresh, sizeof(tresh));
        }
        else
        {
            tresh = 0.0;
            writeMemory(&tresh, sizeof(tresh));
        }
        
        writeMemory(&ip, sizeof(ip)); // write ip = 1;
        readMemory(&ip, sizeof(ip));  //read ip  n for comparison.
        readMemory(&n, sizeof(n));
        for (ip = 1; ip <= n - 1; ip++)
        {
            
            readMemory(&ip, sizeof(ip));   // read ip for comparison
            writeMemory(&iq, sizeof(iq));  // write iq = ip + 1
            readMemory(&iq, sizeof(iq));   // read iq & n for comparison. Keep iq in cpu reg
            readMemory(&n, sizeof(n));
            for (iq = ip + 1; iq <= n; iq++)
            {
                
                readMemory(&ip, sizeof(ip));  // read ip for comparison. Keep ip in cpu reg.
                readMemory(&a[ip][iq], sizeof(a[ip][iq]));  // iq is in cpu reg
                g = 100.0*fabs(a[ip][iq]);
                writeMemory(&g, sizeof(g));
                
                readMemory(&i, sizeof(i));
                readMemory(&d[ip], sizeof(d[ip]));  // ip & iq are in cpu reg.
                readMemory(&d[iq], sizeof(d[iq]));
                readMemory(&g, sizeof(g));
                if (i>4 && (double)(fabs(d[ip]) + g) == (double)fabs(d[ip]) && (double)(fabs(d[iq]) + g) == (double)fabs(d[iq]))
                {
                    // ip & iq are in cpu reg.
                    writeMemory(&a[ip][iq], sizeof(a[ip][iq]));
                    a[ip][iq] = 0.0;
                }
                //** at this point we will remove ip & iq from cpu reg, as cpu regs can notbe kept occupied for too long.
                
                else if (fabs(a[ip][iq])>tresh)
                {
                    readMemory(&ip, sizeof(ip));                // Read FOR ELSE IF CONDITION
                    readMemory(&iq, sizeof(iq));                    // storing ip and iq in cpu reg
                    readMemory(&a[ip][iq], sizeof(a[ip][iq]));
                    readMemory(&tresh, sizeof(tresh));
                    
                    
                    readMemory(&d[ip], sizeof(d[ip]));   // ip and iq already in cpu reg
                    readMemory(&d[iq], sizeof(d[iq]));
                    readMemory(&h, sizeof(h));
                    h = d[iq] - d[ip];
                    
                    readMemory(&h, sizeof(h));      // h in cpu reg
                    readMemory(&g, sizeof(g));
                    if ((double)(fabs(h) + g) == (double)fabs(h))
                    {
                        // h in cpu reg
                        readMemory(&ip, sizeof(ip));
                        readMemory(&iq, sizeof(iq));
                        readMemory(&a[ip][iq], sizeof(a[ip][iq]));
                        t = (a[ip][iq]) / h;
                        writeMemory(&t, sizeof(t));
                    }
                    else
                    {
                        readMemory(&ip, sizeof(ip));
                        readMemory(&iq, sizeof(iq));
                        readMemory(&h, sizeof(h));
                        readMemory(&a[ip][iq], sizeof(a[ip][iq]));
                        theta = 0.5*h / (a[ip][iq]);
                        writeMemory(&theta, sizeof(theta));
                        
                        readMemory(&theta, sizeof(theta));
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta*theta));
                        writeMemory(&t, sizeof(t));
                        
                        readMemory(&theta, sizeof(theta));
                        if (theta<0.0)
                        {
                            readMemory(&t, sizeof(t));
                            t = -t;
                            writeMemory(&t, sizeof(t));
                        }
                    }
                    readMemory(&t, sizeof(t));
                    c = 1.0 / sqrt(1 + t*t);
                    writeMemory(&c, sizeof(c));
                    
                    readMemory(&c, sizeof(c));     // put c in cpu reg
                    readMemory(&t, sizeof(t));
                    s = t*c;
                    writeMemory(&s, sizeof(s));
                    
                    
                    readMemory(&s, sizeof(s));      // c is already in cpu reg
                    tau = s / (1.0 + c);
                    writeMemory(&tau, sizeof(tau));
                    
                    readMemory(&ip, sizeof(ip));      // storing ip and iq in cpu reg
                    readMemory(&iq, sizeof(iq));
                    readMemory(&a[ip][iq], sizeof(a[ip][iq]));
                    readMemory(&t, sizeof(t));
                    h = t*a[ip][iq];
                    writeMemory(&h, sizeof(h));
                    
                    readMemory(&h, sizeof(h));
                    readMemory(&z[ip], sizeof(z[ip]));    // ip in cpu reg
                    z[ip] = z[ip] - h;
                    writeMemory(&z[ip], sizeof(z[ip]));
                    
                    readMemory(&h, sizeof(h));
                    readMemory(&z[iq], sizeof(z[iq]));        // iq in cpu reg
                    z[iq] = z[iq] + h;
                    writeMemory(&z[iq], sizeof(z[iq]));
                    
                    readMemory(&h, sizeof(h));
                    readMemory(&d[ip], sizeof(d[ip]));   // ip in cpu reg
                    d[ip] = d[ip] - h;
                    writeMemory(&d[ip], sizeof(d[ip]));
                    
                    readMemory(&h, sizeof(h));
                    readMemory(&d[iq], sizeof(d[iq]));   // iq in cpu reg
                    d[iq] = d[iq] + h;
                    writeMemory(&d[iq], sizeof(d[iq]));
                    
                    writeMemory(&a[ip][iq], sizeof(a[ip][iq]));    // ip and iq in cpu reg
                    a[ip][iq] = 0.0;
                    
                    writeMemory(&j, sizeof(j));     // write j
                    readMemory(&j, sizeof(j));         // j and ip for comparison
                    readMemory(&ip, sizeof(ip));
                    for (j = 1; j <= ip - 1; j++)
                    {
                        readMemory(&a, sizeof(a));
                        readMemory(&j, sizeof(j));
                        readMemory(&ip, sizeof(ip));
                        readMemory(&iq, sizeof(iq));
                        ROTATE(a, j, ip, j, iq)
                        readMemory(&j, sizeof(j));
                        readMemory(&ip, sizeof(ip));
                        writeMemory(&j, sizeof(j));
                    }
                    writeMemory(&j, sizeof(j));
                    readMemory(&ip, sizeof(ip));
                    readMemory(&j, sizeof(j));
                    readMemory(&iq, sizeof(ip));
                    for (j = ip + 1; j <= iq - 1; j++)
                    {
                        readMemory(&a, sizeof(a));
                        readMemory(&ip, sizeof(ip));
                        readMemory(&j, sizeof(j));
                        readMemory(&j, sizeof(j));
                        readMemory(&iq, sizeof(iq));
                        ROTATE(a, ip, j, j, iq)
                        writeMemory(&a, sizeof(a));
                        writeMemory(&h, sizeof(h));
                        writeMemory(&g, sizeof(g));
                        
                        writeMemory(&j, sizeof(j));       // store j++
                        
                    }
                    writeMemory(&j, sizeof(j));
                    readMemory(&iq, sizeof(iq));
                    readMemory(&j, sizeof(j));
                    readMemory(&n, sizeof(n));
                    for (j = iq + 1; j <= n; j++)
                    {
                        readMemory(&a, sizeof(a));
                        readMemory(&ip, sizeof(ip));
                        readMemory(&j, sizeof(j));
                        readMemory(&iq, sizeof(iq));
                        ROTATE(a, ip, j, iq, j)
                        writeMemory(&a, sizeof(a));
                        writeMemory(&h, sizeof(h));
                        writeMemory(&g, sizeof(g));
                        writeMemory(&j, sizeof(j));     // store j++
                    }
                    writeMemory(&j, sizeof(j));
                    readMemory(&j, sizeof(j));
                    readMemory(&n, sizeof(n));
                    for (j = 1; j <= n; j++)
                    {
                        readMemory(&v, sizeof(v));
                        readMemory(&j, sizeof(j));
                        readMemory(&ip, sizeof(ip));
                        readMemory(&j, sizeof(j));
                        readMemory(&iq, sizeof(iq));
                        ROTATE(v, j, ip, j, iq)
                        writeMemory(&a, sizeof(a));
                        writeMemory(&h, sizeof(h));
                        writeMemory(&g, sizeof(g));
                        writeMemory(&j, sizeof(j));       // store j++
                    }
                    readMemory(&nrot, sizeof(nrot));
                    ++(*nrot);
                    writeMemory(&nrot, sizeof(nrot));
                }
                
                readMemory(&iq ,sizeof(iq));
                readMemory(&n, sizeof(n));
                writeMemory(&iq, sizeof(iq));                // for loop comparison
                
            }
            readMemory(&n, sizeof(n));           // for loop comparison
            readMemory(&ip, sizeof(ip));
            writeMemory(&ip, sizeof(ip));
        }
        
        
        
        
        writeMemory(&ip, sizeof(ip));
        readMemory(&ip, sizeof(ip));
        readMemory(&n, sizeof(n));
        for (ip = 1; ip <= n; ip++)
        {
            readMemory(&ip, sizeof(ip));               // read ip and put it in cpu reg
            readMemory(&z[ip], sizeof(z[ip]));
            readMemory(&b[ip], sizeof(b[ip]));
            writeMemory(&b[ip], sizeof(b[ip]));
            b[ip] = b[ip] + z[ip];
            
            readMemory(&b[ip], sizeof(b[ip]));            // ip already in cpu reg
            writeMemory(&d[ip], sizeof(d[ip]));
            d[ip] = b[ip];
            
            writeMemory(&z[ip], sizeof(z[ip]));            // ip already in cpu reg
            z[ip] = 0.0;
            
            
            readMemory(&n, sizeof(n));                    // ip already in cpu reg
            writeMemory(&ip, sizeof(ip));                // write ip++
        }
        
        readMemory(&i, sizeof(i));                // read i for comparison
        writeMemory(&i, sizeof(i));                // write i++
    }
    nrerror("Too many iterations in routine jacobi");
    
}


//double x[16384];
//__declspec(align(4)) x[16384];
//__declspec(align(4)) uint32_t x[8192];
//#pragma align(4)
uint32_t i, j, y;



void mapBL()
{
    if(bl==0) {CURR_B_LEN = 1;}
    if(bl==1) {CURR_B_LEN = 2;}
    if(bl==2) {CURR_B_LEN = 4;}
    if(bl==3) {CURR_B_LEN = 8;}
}

void mapN()
{
    if(set_size==0) { CURR_WAY = 1;}
    if(set_size==1) { CURR_WAY = 2;}
    if(set_size==2) { CURR_WAY = 4;}
    if(set_size==3) { CURR_WAY = 8;}
    if(set_size==4) { CURR_WAY = 16;}
}
    
int main()
{
   /* FILE* pFile =  fopen("/Users/nihit/Desktop/first1.txt", "w");
    fputs("Write_Strategy\n 0: WRITE BACK, 1: WRITE THROUGH ALLOCATE; 2: WRITE ALLOCATE\n",pFile);
    fputs("WriteStrategy\tSet_Size(N)\tBurst Length(L)\tcntr_rd_mem\tcntr_rd_line\tcntr_rd_miss\tcntr_rd_d_replace\tcntr_rd_cache\tcntr_rd_mem_fun\n",pFile);
    fclose(pFile);*/
    
    init_cache();
    
    //printf("Please Enter Write Strategy(WR_BACK : 0, WR_ALLOC : 1, WR_NON_ALLOC    : 2) ; No of ways  & Burst Length\n");
    //scanf("%u %u %u", &CURR_WR_STRATEGY, &CURR_WAY, &CURR_B_LEN);
    
    
    
    for(int wr_s=0; wr_s < 3; wr_s++)
    {
        for(bl=0; bl < 4; bl++)
        {
            for(set_size=0; set_size < 5; set_size++)
            {
                mapBL();
                mapN();
                block_offset_bits_cnt = get_Offset_Bits_Count(CURR_B_LEN);
                CURR_LINES_CNT = get_Number_of_Lines(CURR_WAY, CURR_B_LEN);
                line_bits_cnt = get_Line_Bits_Count(CURR_LINES_CNT);
                tag_bits_cnt = 32 - (block_offset_bits_cnt + line_bits_cnt);
                
                
               
                for (int j =1; j<128; j++)
                {
                    for(int k =1; k<128; k++)
                    {
                        a[j][k] = 2;                            // symmetric matrix
                    }
                }
                double **v = dmatrix(1,128,1,128);               // allocates double matrix with subscript from 1 to 128.
                double *d =  dvector(1,128);                     //  allocates double matrix with subscript range from 1
                double **a = dmatrix(1, 128, 1, 128);          // returns pointer to 2-D double matrix
                
               
                jacobi(a, 128, d, v, &nrot);
                
                FILE* pFile = fopen("/Users/nihit/Desktop/first1.txt", "a");
                
              /*
               
               
               
                
               
                 */
                
                
                //fprintf(pFile,"%d\t\n",wr_s);
                //fprintf(pFile,"%d\t",CURR_WAY);
                //fprintf(pFile,"%d\t", CURR_B_LEN);
                fprintf(pFile,"%d\t\n",cntr_wr_mem);
                fprintf(pFile,"%d\t",cntr_wr_line);
                fprintf(pFile,"%d\t",cntr_wr_miss);
                fprintf(pFile,"%d\t",cntr_wr_cache);
                fprintf(pFile,"%d\t",cntr_wr_mem_fun);
                fprintf(pFile,"%d\t",cntr_wr_d_replace);
                fprintf(pFile,"%d\t",cntr_wr_through_mem);
                
                
                fprintf(pFile,"%d\t",cntr_rd_mem);
                fprintf(pFile,"%d\t",cntr_rd_line);
                fprintf(pFile,"%d\t",cntr_rd_miss);
                fprintf(pFile,"%d\t",cntr_rd_cache);
                fprintf(pFile,"%d\t",cntr_rd_mem_fun);
                fprintf(pFile,"%d\t",cntr_rd_d_replace);
                
                fclose(pFile);
                
                
                printf("cntr_wr_mem : %d \n", cntr_wr_mem);
                printf("cntr_rd_mem : %d \n", cntr_rd_mem);
                printf("cntr_wr_line : %d \n", cntr_wr_line);
                printf("cntr_rd_line : %d \n", cntr_rd_line);
                printf("cntr_wr_miss : %d \n", cntr_wr_miss);
                printf("cntr_rd_miss : %d \n", cntr_rd_miss);
                printf("cntr_wr_d_replace : %d \n", cntr_wr_d_replace);
                printf("cntr_rd_d_replace : %d \n", cntr_rd_d_replace);
                printf("cntr_wr_cache : %d \n", cntr_wr_cache);
                printf("cntr_rd_cache : %d \n", cntr_rd_cache);
                printf("cntr_wr_through_mem : %d \n", cntr_wr_through_mem);
                printf("cntr_dirty : %d \n", cntr_dirty);
                printf("cntr_wr_mem_fun : %d \n", cntr_wr_mem_fun);
                printf("cntr_rd_mem_fun : %d \n", cntr_rd_mem_fun);
                init_cache();
                
                
            }
            
            
        }
        
    }
    
    getchar();
    getchar();
    return 0;
}





