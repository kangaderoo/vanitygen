/********************************************************************\
 *
 *      FILE:     rmd160.h
 *
 *      CONTENTS: Header file for a sample C-implementation of the
 *                RIPEMD-160 hash-function. 
 *      TARGET:   any computer with an ANSI C compiler
 *
 *      AUTHOR:   Antoon Bosselaers, ESAT-COSIC
 *      DATE:     1 March 1996
 *      VERSION:  1.0
 *
 *      Copyright (c) Katholieke Universiteit Leuven
 *      1996, All Rights Reserved
 *
\********************************************************************/

#ifndef  RMD160H           /* make sure this file is read only once */
#define  RMD160H

/********************************************************************/
#include <immintrin.h>
#include <inttypes.h>
/* typedef 8 and 32 bit types, resp.  */
/* adapt these, if necessary, 
   for your operating system and compiler */
typedef    unsigned char        byte;
typedef    unsigned int         dword;

/* if this line causes a compiler error, 
   adapt the defintion of dword above */
typedef int the_correct_size_was_chosen [sizeof (dword) == 4? 1: -1];

/********************************************************************/
static const uint32_t mdbuf_quad[5*4] = {
		0x67452301, 0x67452301, 0x67452301, 0x67452301,
		0xefcdab89, 0xefcdab89, 0xefcdab89, 0xefcdab89,
		0x98badcfe, 0x98badcfe, 0x98badcfe, 0x98badcfe,
		0x10325476, 0x10325476, 0x10325476, 0x10325476,
		0xc3d2e1f0, 0xc3d2e1f0, 0xc3d2e1f0, 0xc3d2e1f0
};

/* macro definitions */

/* collect four bytes into one word: */
#define BYTES_TO_DWORD(strptr)                    \
            (((dword) *((strptr)+3) << 24) | \
             ((dword) *((strptr)+2) << 16) | \
             ((dword) *((strptr)+1) <<  8) | \
             ((dword) *(strptr)))

/* ROL(x, n) cyclically rotates x over n bits to the left */
/* x must be of an unsigned 32 bits type and 0 <= n < 32. */
#define ROL(x, n)        (((x) << (n)) | ((x) >> (32-(n))))

/* the five basic functions F(), G() and H() */
#define F(x, y, z)        ((x) ^ (y) ^ (z)) 
#define G(x, y, z)        (((x) & (y)) | (~(x) & (z))) 
#define H(x, y, z)        (((x) | ~(y)) ^ (z))
#define I(x, y, z)        (((x) & (z)) | ((y) & ~(z))) 
#define J(x, y, z)        ((x) ^ ((y) | ~(z)))
  
/* the ten basic operations FF() through III() */
#define FF(a, b, c, d, e, x, s)        {\
      (a) += F((b), (c), (d)) + (x);\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define GG(a, b, c, d, e, x, s)        {\
      (a) += G((b), (c), (d)) + (x) + 0x5a827999UL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define HH(a, b, c, d, e, x, s)        {\
      (a) += H((b), (c), (d)) + (x) + 0x6ed9eba1UL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define II(a, b, c, d, e, x, s)        {\
      (a) += I((b), (c), (d)) + (x) + 0x8f1bbcdcUL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define JJ(a, b, c, d, e, x, s)        {\
      (a) += J((b), (c), (d)) + (x) + 0xa953fd4eUL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define FFF(a, b, c, d, e, x, s)        {\
      (a) += F((b), (c), (d)) + (x);\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define GGG(a, b, c, d, e, x, s)        {\
      (a) += G((b), (c), (d)) + (x) + 0x7a6d76e9UL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define HHH(a, b, c, d, e, x, s)        {\
      (a) += H((b), (c), (d)) + (x) + 0x6d703ef3UL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define III(a, b, c, d, e, x, s)        {\
      (a) += I((b), (c), (d)) + (x) + 0x5c4dd124UL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }
#define JJJ(a, b, c, d, e, x, s)        {\
      (a) += J((b), (c), (d)) + (x) + 0x50a28be6UL;\
      (a) = ROL((a), (s)) + (e);\
      (c) = ROL((c), 10);\
   }

/********************************************************************/

/* function prototypes */

void MDinit(dword *MDbuf);
/*
 *  initializes MDbuffer to "magic constants"
 */

void MM_MDinit(uint32_t *MDbuf);
/*
 *  initializes MDbuffer to "magic constants"
 */

void compress(dword *MDbuf, dword *X);
/*
 *  the compression function.
 *  transforms MDbuf using message bytes X[0] through X[15]
 */
void MM_compress(__m128i *MDbuf, __m128i *X);

void MDfinish(dword *MDbuf, byte *strptr, dword lswlen, dword mswlen);
/*
 *  puts bytes from strptr into X and pad out; appends length 
 *  and finally, compresses the last block(s)
 *  note: length in bits == 8 * (lswlen + 2^32 mswlen).
 *  note: there are (lswlen mod 64) bytes left in strptr.
 */
void MM_MDfinish(__m128i *MDbuf, __m128i *strptr, dword lswlen, dword mswlen);


void MM_matrix_transpose_r2c(__m128i* inbuf,__m128i* outbuf, uint32_t rows, uint32_t colums);
/*
 * uses uint32_t array as input and transposes the rows to colunms
 */

void MM_matrix_transpose_c2r(__m128i* inbuf,__m128i* outbuf, uint32_t rows, uint32_t colums);
/*
 * uses uint32_t array as input and transposes the columns to rows
 */

#endif  /* RMD160H */

/*********************** end of file rmd160.h ***********************/

