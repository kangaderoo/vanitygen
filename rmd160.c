/********************************************************************\
 *
 *      FILE:     rmd160.c
 *
 *      CONTENTS: A sample C-implementation of the RIPEMD-160
 *                hash-function.
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

/*  header files */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rmd160.h"      
#include <immintrin.h>
#include <inttypes.h>
/********************************************************************/

void MDinit(dword *MDbuf)
{
   MDbuf[0] = 0x67452301UL;
   MDbuf[1] = 0xefcdab89UL;
   MDbuf[2] = 0x98badcfeUL;
   MDbuf[3] = 0x10325476UL;
   MDbuf[4] = 0xc3d2e1f0UL;

//   return;
}

void MM_MDinit(uint32_t *MDbuf)
{
	__m128i *ConstPrt = (__m128i*) mdbuf_quad;
	__m128i *BufPrt = (__m128i*) MDbuf;
	uint32_t i;
	for(i=0;i<5;i++)
		BufPrt[i] = ConstPrt[i];
#if 0
   MDbuf[0] = 0x67452301UL;
   MDbuf[4] = 0xefcdab89UL;
   MDbuf[8] = 0x98badcfeUL;
   MDbuf[12] = 0x10325476UL;
   MDbuf[16] = 0xc3d2e1f0UL;

   MDbuf[1] = 0x67452301UL;
   MDbuf[5] = 0xefcdab89UL;
   MDbuf[9] = 0x98badcfeUL;
   MDbuf[13] = 0x10325476UL;
   MDbuf[17] = 0xc3d2e1f0UL;

   MDbuf[2] = 0x67452301UL;
   MDbuf[6] = 0xefcdab89UL;
   MDbuf[10] = 0x98badcfeUL;
   MDbuf[14] = 0x10325476UL;
   MDbuf[18] = 0xc3d2e1f0UL;

   MDbuf[3] = 0x67452301UL;
   MDbuf[7] = 0xefcdab89UL;
   MDbuf[11] = 0x98badcfeUL;
   MDbuf[15] = 0x10325476UL;
   MDbuf[19] = 0xc3d2e1f0UL;
#endif
//   return;
}


static inline __m128i _mm_ROL(__m128i x, int n) {
    return _mm_slli_epi32(x, n) | _mm_srli_epi32(x, 32 - n);
}

/* the five basic functions F(), G(), H(), I() and J() */
#define F(x, y, z)        ((x) ^ (y) ^ (z))
#define G(x, y, z)        (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z)        (((x) | ~(y)) ^ (z))
#define I(x, y, z)        (((x) & (z)) | ((y) & ~(z)))
#define J(x, y, z)        ((x) ^ ((y) | ~(z)))

static inline __m128i _mm_F(__m128i x, __m128i y, __m128i z) {
    return ((x)^(y)^(z));
}

static inline __m128i _mm_G(__m128i x, __m128i y, __m128i z) {
    return (((x)&(y))|(~(x)&(z)));
}

static inline __m128i _mm_H(__m128i x, __m128i y, __m128i z) {
    return (((x)|~(y))^(z));
}

static inline __m128i _mm_I(__m128i x, __m128i y, __m128i z) {
    return (((x)&(z))|((y)&~(z)));
}

static inline __m128i _mm_J(__m128i x, __m128i y, __m128i z) {
    return ((x)^((y)|~(z)));
}

static const uint32_t GG_const_arr[4] __attribute__((aligned(16))) = {0x5a827999UL,0x5a827999UL,0x5a827999UL,0x5a827999UL};
static const uint32_t HH_const_arr[4] __attribute__((aligned(16))) = {0x6ed9eba1UL,0x6ed9eba1UL,0x6ed9eba1UL,0x6ed9eba1UL};
static const uint32_t II_const_arr[4] __attribute__((aligned(16))) = {0x8f1bbcdcUL,0x8f1bbcdcUL,0x8f1bbcdcUL,0x8f1bbcdcUL};
static const uint32_t JJ_const_arr[4] __attribute__((aligned(16))) = {0xa953fd4eUL,0xa953fd4eUL,0xa953fd4eUL,0xa953fd4eUL};
static const __m128i *GG_const = (__m128i*)GG_const_arr;
static const __m128i *HH_const = (__m128i*)HH_const_arr;
static const __m128i *II_const = (__m128i*)II_const_arr;
static const __m128i *JJ_const = (__m128i*)JJ_const_arr;


static inline void _mm_FF(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_F((b),*(c),(d)),(x)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_GG(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_G((b),*(c),(d)),_mm_add_epi32((x),*GG_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_HH(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_H((b),*(c),(d)),_mm_add_epi32((x),*HH_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_II(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_I((b),*(c),(d)),_mm_add_epi32((x),*II_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_JJ(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_J((b),*(c),(d)),_mm_add_epi32((x),*JJ_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}

static const uint32_t GGG_const_arr[4] __attribute__((aligned(16))) = {0x7a6d76e9UL,0x7a6d76e9UL,0x7a6d76e9UL,0x7a6d76e9UL};
static const uint32_t HHH_const_arr[4] __attribute__((aligned(16))) = {0x6d703ef3UL,0x6d703ef3UL,0x6d703ef3UL,0x6d703ef3UL};
static const uint32_t III_const_arr[4] __attribute__((aligned(16))) = {0x5c4dd124UL,0x5c4dd124UL,0x5c4dd124UL,0x5c4dd124UL};
static const uint32_t JJJ_const_arr[4] __attribute__((aligned(16))) = {0x50a28be6UL,0x50a28be6UL,0x50a28be6UL,0x50a28be6UL};
static const __m128i *GGG_const = (__m128i*)GGG_const_arr;
static const __m128i *HHH_const = (__m128i*)HHH_const_arr;
static const __m128i *III_const = (__m128i*)III_const_arr;
static const __m128i *JJJ_const = (__m128i*)JJJ_const_arr;


static inline void _mm_FFF(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_F((b),*(c),(d)),(x)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_GGG(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_G((b),*(c),(d)),_mm_add_epi32((x),*GGG_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_HHH(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_H((b),*(c),(d)),_mm_add_epi32((x),*HHH_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_III(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_I((b),*(c),(d)),_mm_add_epi32((x),*III_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}
static inline void _mm_JJJ(__m128i *a, __m128i b, __m128i *c, __m128i d, __m128i e, __m128i x, int s) {
	*(a) = _mm_add_epi32(*(a),_mm_add_epi32(_mm_J((b),*(c),(d)),_mm_add_epi32((x),*JJJ_const)));
	*(c) = _mm_ROL(*(c),10);
	*(a) = _mm_add_epi32(_mm_ROL(*(a),s),(e));
}

/********************************************************************/

void compress(dword *MDbuf, dword *X)
{
   dword aa = MDbuf[0],  bb = MDbuf[1],  cc = MDbuf[2],
         dd = MDbuf[3],  ee = MDbuf[4];
   dword aaa = MDbuf[0], bbb = MDbuf[1], ccc = MDbuf[2],
         ddd = MDbuf[3], eee = MDbuf[4];

   /* round 1 */
   FF(aa, bb, cc, dd, ee, X[ 0], 11);
   FF(ee, aa, bb, cc, dd, X[ 1], 14);
   FF(dd, ee, aa, bb, cc, X[ 2], 15);
   FF(cc, dd, ee, aa, bb, X[ 3], 12);
   FF(bb, cc, dd, ee, aa, X[ 4],  5);
   FF(aa, bb, cc, dd, ee, X[ 5],  8);
   FF(ee, aa, bb, cc, dd, X[ 6],  7);
   FF(dd, ee, aa, bb, cc, X[ 7],  9);
   FF(cc, dd, ee, aa, bb, X[ 8], 11);
   FF(bb, cc, dd, ee, aa, X[ 9], 13);
   FF(aa, bb, cc, dd, ee, X[10], 14);
   FF(ee, aa, bb, cc, dd, X[11], 15);
   FF(dd, ee, aa, bb, cc, X[12],  6);
   FF(cc, dd, ee, aa, bb, X[13],  7);
   FF(bb, cc, dd, ee, aa, X[14],  9);
   FF(aa, bb, cc, dd, ee, X[15],  8);
                             
   /* round 2 */
   GG(ee, aa, bb, cc, dd, X[ 7],  7);
   GG(dd, ee, aa, bb, cc, X[ 4],  6);
   GG(cc, dd, ee, aa, bb, X[13],  8);
   GG(bb, cc, dd, ee, aa, X[ 1], 13);
   GG(aa, bb, cc, dd, ee, X[10], 11);
   GG(ee, aa, bb, cc, dd, X[ 6],  9);
   GG(dd, ee, aa, bb, cc, X[15],  7);
   GG(cc, dd, ee, aa, bb, X[ 3], 15);
   GG(bb, cc, dd, ee, aa, X[12],  7);
   GG(aa, bb, cc, dd, ee, X[ 0], 12);
   GG(ee, aa, bb, cc, dd, X[ 9], 15);
   GG(dd, ee, aa, bb, cc, X[ 5],  9);
   GG(cc, dd, ee, aa, bb, X[ 2], 11);
   GG(bb, cc, dd, ee, aa, X[14],  7);
   GG(aa, bb, cc, dd, ee, X[11], 13);
   GG(ee, aa, bb, cc, dd, X[ 8], 12);

   /* round 3 */
   HH(dd, ee, aa, bb, cc, X[ 3], 11);
   HH(cc, dd, ee, aa, bb, X[10], 13);
   HH(bb, cc, dd, ee, aa, X[14],  6);
   HH(aa, bb, cc, dd, ee, X[ 4],  7);
   HH(ee, aa, bb, cc, dd, X[ 9], 14);
   HH(dd, ee, aa, bb, cc, X[15],  9);
   HH(cc, dd, ee, aa, bb, X[ 8], 13);
   HH(bb, cc, dd, ee, aa, X[ 1], 15);
   HH(aa, bb, cc, dd, ee, X[ 2], 14);
   HH(ee, aa, bb, cc, dd, X[ 7],  8);
   HH(dd, ee, aa, bb, cc, X[ 0], 13);
   HH(cc, dd, ee, aa, bb, X[ 6],  6);
   HH(bb, cc, dd, ee, aa, X[13],  5);
   HH(aa, bb, cc, dd, ee, X[11], 12);
   HH(ee, aa, bb, cc, dd, X[ 5],  7);
   HH(dd, ee, aa, bb, cc, X[12],  5);

   /* round 4 */
   II(cc, dd, ee, aa, bb, X[ 1], 11);
   II(bb, cc, dd, ee, aa, X[ 9], 12);
   II(aa, bb, cc, dd, ee, X[11], 14);
   II(ee, aa, bb, cc, dd, X[10], 15);
   II(dd, ee, aa, bb, cc, X[ 0], 14);
   II(cc, dd, ee, aa, bb, X[ 8], 15);
   II(bb, cc, dd, ee, aa, X[12],  9);
   II(aa, bb, cc, dd, ee, X[ 4],  8);
   II(ee, aa, bb, cc, dd, X[13],  9);
   II(dd, ee, aa, bb, cc, X[ 3], 14);
   II(cc, dd, ee, aa, bb, X[ 7],  5);
   II(bb, cc, dd, ee, aa, X[15],  6);
   II(aa, bb, cc, dd, ee, X[14],  8);
   II(ee, aa, bb, cc, dd, X[ 5],  6);
   II(dd, ee, aa, bb, cc, X[ 6],  5);
   II(cc, dd, ee, aa, bb, X[ 2], 12);

   /* round 5 */
   JJ(bb, cc, dd, ee, aa, X[ 4],  9);
   JJ(aa, bb, cc, dd, ee, X[ 0], 15);
   JJ(ee, aa, bb, cc, dd, X[ 5],  5);
   JJ(dd, ee, aa, bb, cc, X[ 9], 11);
   JJ(cc, dd, ee, aa, bb, X[ 7],  6);
   JJ(bb, cc, dd, ee, aa, X[12],  8);
   JJ(aa, bb, cc, dd, ee, X[ 2], 13);
   JJ(ee, aa, bb, cc, dd, X[10], 12);
   JJ(dd, ee, aa, bb, cc, X[14],  5);
   JJ(cc, dd, ee, aa, bb, X[ 1], 12);
   JJ(bb, cc, dd, ee, aa, X[ 3], 13);
   JJ(aa, bb, cc, dd, ee, X[ 8], 14);
   JJ(ee, aa, bb, cc, dd, X[11], 11);
   JJ(dd, ee, aa, bb, cc, X[ 6],  8);
   JJ(cc, dd, ee, aa, bb, X[15],  5);
   JJ(bb, cc, dd, ee, aa, X[13],  6);

   /* parallel round 1 */
   JJJ(aaa, bbb, ccc, ddd, eee, X[ 5],  8);
   JJJ(eee, aaa, bbb, ccc, ddd, X[14],  9);
   JJJ(ddd, eee, aaa, bbb, ccc, X[ 7],  9);
   JJJ(ccc, ddd, eee, aaa, bbb, X[ 0], 11);
   JJJ(bbb, ccc, ddd, eee, aaa, X[ 9], 13);
   JJJ(aaa, bbb, ccc, ddd, eee, X[ 2], 15);
   JJJ(eee, aaa, bbb, ccc, ddd, X[11], 15);
   JJJ(ddd, eee, aaa, bbb, ccc, X[ 4],  5);
   JJJ(ccc, ddd, eee, aaa, bbb, X[13],  7);
   JJJ(bbb, ccc, ddd, eee, aaa, X[ 6],  7);
   JJJ(aaa, bbb, ccc, ddd, eee, X[15],  8);
   JJJ(eee, aaa, bbb, ccc, ddd, X[ 8], 11);
   JJJ(ddd, eee, aaa, bbb, ccc, X[ 1], 14);
   JJJ(ccc, ddd, eee, aaa, bbb, X[10], 14);
   JJJ(bbb, ccc, ddd, eee, aaa, X[ 3], 12);
   JJJ(aaa, bbb, ccc, ddd, eee, X[12],  6);

   /* parallel round 2 */
   III(eee, aaa, bbb, ccc, ddd, X[ 6],  9); 
   III(ddd, eee, aaa, bbb, ccc, X[11], 13);
   III(ccc, ddd, eee, aaa, bbb, X[ 3], 15);
   III(bbb, ccc, ddd, eee, aaa, X[ 7],  7);
   III(aaa, bbb, ccc, ddd, eee, X[ 0], 12);
   III(eee, aaa, bbb, ccc, ddd, X[13],  8);
   III(ddd, eee, aaa, bbb, ccc, X[ 5],  9);
   III(ccc, ddd, eee, aaa, bbb, X[10], 11);
   III(bbb, ccc, ddd, eee, aaa, X[14],  7);
   III(aaa, bbb, ccc, ddd, eee, X[15],  7);
   III(eee, aaa, bbb, ccc, ddd, X[ 8], 12);
   III(ddd, eee, aaa, bbb, ccc, X[12],  7);
   III(ccc, ddd, eee, aaa, bbb, X[ 4],  6);
   III(bbb, ccc, ddd, eee, aaa, X[ 9], 15);
   III(aaa, bbb, ccc, ddd, eee, X[ 1], 13);
   III(eee, aaa, bbb, ccc, ddd, X[ 2], 11);

   /* parallel round 3 */
   HHH(ddd, eee, aaa, bbb, ccc, X[15],  9);
   HHH(ccc, ddd, eee, aaa, bbb, X[ 5],  7);
   HHH(bbb, ccc, ddd, eee, aaa, X[ 1], 15);
   HHH(aaa, bbb, ccc, ddd, eee, X[ 3], 11);
   HHH(eee, aaa, bbb, ccc, ddd, X[ 7],  8);
   HHH(ddd, eee, aaa, bbb, ccc, X[14],  6);
   HHH(ccc, ddd, eee, aaa, bbb, X[ 6],  6);
   HHH(bbb, ccc, ddd, eee, aaa, X[ 9], 14);
   HHH(aaa, bbb, ccc, ddd, eee, X[11], 12);
   HHH(eee, aaa, bbb, ccc, ddd, X[ 8], 13);
   HHH(ddd, eee, aaa, bbb, ccc, X[12],  5);
   HHH(ccc, ddd, eee, aaa, bbb, X[ 2], 14);
   HHH(bbb, ccc, ddd, eee, aaa, X[10], 13);
   HHH(aaa, bbb, ccc, ddd, eee, X[ 0], 13);
   HHH(eee, aaa, bbb, ccc, ddd, X[ 4],  7);
   HHH(ddd, eee, aaa, bbb, ccc, X[13],  5);

   /* parallel round 4 */   
   GGG(ccc, ddd, eee, aaa, bbb, X[ 8], 15);
   GGG(bbb, ccc, ddd, eee, aaa, X[ 6],  5);
   GGG(aaa, bbb, ccc, ddd, eee, X[ 4],  8);
   GGG(eee, aaa, bbb, ccc, ddd, X[ 1], 11);
   GGG(ddd, eee, aaa, bbb, ccc, X[ 3], 14);
   GGG(ccc, ddd, eee, aaa, bbb, X[11], 14);
   GGG(bbb, ccc, ddd, eee, aaa, X[15],  6);
   GGG(aaa, bbb, ccc, ddd, eee, X[ 0], 14);
   GGG(eee, aaa, bbb, ccc, ddd, X[ 5],  6);
   GGG(ddd, eee, aaa, bbb, ccc, X[12],  9);
   GGG(ccc, ddd, eee, aaa, bbb, X[ 2], 12);
   GGG(bbb, ccc, ddd, eee, aaa, X[13],  9);
   GGG(aaa, bbb, ccc, ddd, eee, X[ 9], 12);
   GGG(eee, aaa, bbb, ccc, ddd, X[ 7],  5);
   GGG(ddd, eee, aaa, bbb, ccc, X[10], 15);
   GGG(ccc, ddd, eee, aaa, bbb, X[14],  8);

   /* parallel round 5 */
   FFF(bbb, ccc, ddd, eee, aaa, X[12] ,  8);
   FFF(aaa, bbb, ccc, ddd, eee, X[15] ,  5);
   FFF(eee, aaa, bbb, ccc, ddd, X[10] , 12);
   FFF(ddd, eee, aaa, bbb, ccc, X[ 4] ,  9);
   FFF(ccc, ddd, eee, aaa, bbb, X[ 1] , 12);
   FFF(bbb, ccc, ddd, eee, aaa, X[ 5] ,  5);
   FFF(aaa, bbb, ccc, ddd, eee, X[ 8] , 14);
   FFF(eee, aaa, bbb, ccc, ddd, X[ 7] ,  6);
   FFF(ddd, eee, aaa, bbb, ccc, X[ 6] ,  8);
   FFF(ccc, ddd, eee, aaa, bbb, X[ 2] , 13);
   FFF(bbb, ccc, ddd, eee, aaa, X[13] ,  6);
   FFF(aaa, bbb, ccc, ddd, eee, X[14] ,  5);
   FFF(eee, aaa, bbb, ccc, ddd, X[ 0] , 15);
   FFF(ddd, eee, aaa, bbb, ccc, X[ 3] , 13);
   FFF(ccc, ddd, eee, aaa, bbb, X[ 9] , 11);
   FFF(bbb, ccc, ddd, eee, aaa, X[11] , 11);

   /* combine results */
   ddd += cc + MDbuf[1];               /* final result for MDbuf[0] */
   MDbuf[1] = MDbuf[2] + dd + eee;
   MDbuf[2] = MDbuf[3] + ee + aaa;
   MDbuf[3] = MDbuf[4] + aa + bbb;
   MDbuf[4] = MDbuf[0] + bb + ccc;
   MDbuf[0] = ddd;

   return;
}

void MM_compress(__m128i *MDbuf, __m128i *X)
{
   __m128i aa = MDbuf[0],  bb = MDbuf[1],  cc = MDbuf[2],
         dd = MDbuf[3],  ee = MDbuf[4];
   __m128i aaa = MDbuf[0], bbb = MDbuf[1], ccc = MDbuf[2],
         ddd = MDbuf[3], eee = MDbuf[4];

   /* round 1 */
   _mm_FF(&aa, bb, &cc, dd, ee, X[ 0], 11);
   _mm_FF(&ee, aa, &bb, cc, dd, X[ 1], 14);
   _mm_FF(&dd, ee, &aa, bb, cc, X[ 2], 15);
   _mm_FF(&cc, dd, &ee, aa, bb, X[ 3], 12);
   _mm_FF(&bb, cc, &dd, ee, aa, X[ 4],  5);
   _mm_FF(&aa, bb, &cc, dd, ee, X[ 5],  8);
   _mm_FF(&ee, aa, &bb, cc, dd, X[ 6],  7);
   _mm_FF(&dd, ee, &aa, bb, cc, X[ 7],  9);
   _mm_FF(&cc, dd, &ee, aa, bb, X[ 8], 11);
   _mm_FF(&bb, cc, &dd, ee, aa, X[ 9], 13);
   _mm_FF(&aa, bb, &cc, dd, ee, X[10], 14);
   _mm_FF(&ee, aa, &bb, cc, dd, X[11], 15);
   _mm_FF(&dd, ee, &aa, bb, cc, X[12],  6);
   _mm_FF(&cc, dd, &ee, aa, bb, X[13],  7);
   _mm_FF(&bb, cc, &dd, ee, aa, X[14],  9);
   _mm_FF(&aa, bb, &cc, dd, ee, X[15],  8);

   /* round 2 */
   _mm_GG(&ee, aa, &bb, cc, dd, X[ 7],  7);
   _mm_GG(&dd, ee, &aa, bb, cc, X[ 4],  6);
   _mm_GG(&cc, dd, &ee, aa, bb, X[13],  8);
   _mm_GG(&bb, cc, &dd, ee, aa, X[ 1], 13);
   _mm_GG(&aa, bb, &cc, dd, ee, X[10], 11);
   _mm_GG(&ee, aa, &bb, cc, dd, X[ 6],  9);
   _mm_GG(&dd, ee, &aa, bb, cc, X[15],  7);
   _mm_GG(&cc, dd, &ee, aa, bb, X[ 3], 15);
   _mm_GG(&bb, cc, &dd, ee, aa, X[12],  7);
   _mm_GG(&aa, bb, &cc, dd, ee, X[ 0], 12);
   _mm_GG(&ee, aa, &bb, cc, dd, X[ 9], 15);
   _mm_GG(&dd, ee, &aa, bb, cc, X[ 5],  9);
   _mm_GG(&cc, dd, &ee, aa, bb, X[ 2], 11);
   _mm_GG(&bb, cc, &dd, ee, aa, X[14],  7);
   _mm_GG(&aa, bb, &cc, dd, ee, X[11], 13);
   _mm_GG(&ee, aa, &bb, cc, dd, X[ 8], 12);

   /* round 3 */
   _mm_HH(&dd, ee, &aa, bb, cc, X[ 3], 11);
   _mm_HH(&cc, dd, &ee, aa, bb, X[10], 13);
   _mm_HH(&bb, cc, &dd, ee, aa, X[14],  6);
   _mm_HH(&aa, bb, &cc, dd, ee, X[ 4],  7);
   _mm_HH(&ee, aa, &bb, cc, dd, X[ 9], 14);
   _mm_HH(&dd, ee, &aa, bb, cc, X[15],  9);
   _mm_HH(&cc, dd, &ee, aa, bb, X[ 8], 13);
   _mm_HH(&bb, cc, &dd, ee, aa, X[ 1], 15);
   _mm_HH(&aa, bb, &cc, dd, ee, X[ 2], 14);
   _mm_HH(&ee, aa, &bb, cc, dd, X[ 7],  8);
   _mm_HH(&dd, ee, &aa, bb, cc, X[ 0], 13);
   _mm_HH(&cc, dd, &ee, aa, bb, X[ 6],  6);
   _mm_HH(&bb, cc, &dd, ee, aa, X[13],  5);
   _mm_HH(&aa, bb, &cc, dd, ee, X[11], 12);
   _mm_HH(&ee, aa, &bb, cc, dd, X[ 5],  7);
   _mm_HH(&dd, ee, &aa, bb, cc, X[12],  5);

   /* round 4 */
   _mm_II(&cc, dd, &ee, aa, bb, X[ 1], 11);
   _mm_II(&bb, cc, &dd, ee, aa, X[ 9], 12);
   _mm_II(&aa, bb, &cc, dd, ee, X[11], 14);
   _mm_II(&ee, aa, &bb, cc, dd, X[10], 15);
   _mm_II(&dd, ee, &aa, bb, cc, X[ 0], 14);
   _mm_II(&cc, dd, &ee, aa, bb, X[ 8], 15);
   _mm_II(&bb, cc, &dd, ee, aa, X[12],  9);
   _mm_II(&aa, bb, &cc, dd, ee, X[ 4],  8);
   _mm_II(&ee, aa, &bb, cc, dd, X[13],  9);
   _mm_II(&dd, ee, &aa, bb, cc, X[ 3], 14);
   _mm_II(&cc, dd, &ee, aa, bb, X[ 7],  5);
   _mm_II(&bb, cc, &dd, ee, aa, X[15],  6);
   _mm_II(&aa, bb, &cc, dd, ee, X[14],  8);
   _mm_II(&ee, aa, &bb, cc, dd, X[ 5],  6);
   _mm_II(&dd, ee, &aa, bb, cc, X[ 6],  5);
   _mm_II(&cc, dd, &ee, aa, bb, X[ 2], 12);

   /* round 5 */
   _mm_JJ(&bb, cc, &dd, ee, aa, X[ 4],  9);
   _mm_JJ(&aa, bb, &cc, dd, ee, X[ 0], 15);
   _mm_JJ(&ee, aa, &bb, cc, dd, X[ 5],  5);
   _mm_JJ(&dd, ee, &aa, bb, cc, X[ 9], 11);
   _mm_JJ(&cc, dd, &ee, aa, bb, X[ 7],  6);
   _mm_JJ(&bb, cc, &dd, ee, aa, X[12],  8);
   _mm_JJ(&aa, bb, &cc, dd, ee, X[ 2], 13);
   _mm_JJ(&ee, aa, &bb, cc, dd, X[10], 12);
   _mm_JJ(&dd, ee, &aa, bb, cc, X[14],  5);
   _mm_JJ(&cc, dd, &ee, aa, bb, X[ 1], 12);
   _mm_JJ(&bb, cc, &dd, ee, aa, X[ 3], 13);
   _mm_JJ(&aa, bb, &cc, dd, ee, X[ 8], 14);
   _mm_JJ(&ee, aa, &bb, cc, dd, X[11], 11);
   _mm_JJ(&dd, ee, &aa, bb, cc, X[ 6],  8);
   _mm_JJ(&cc, dd, &ee, aa, bb, X[15],  5);
   _mm_JJ(&bb, cc, &dd, ee, aa, X[13],  6);

   /* parallel round 1 */
   _mm_JJJ(&aaa, bbb, &ccc, ddd, eee, X[ 5],  8);
   _mm_JJJ(&eee, aaa, &bbb, ccc, ddd, X[14],  9);
   _mm_JJJ(&ddd, eee, &aaa, bbb, ccc, X[ 7],  9);
   _mm_JJJ(&ccc, ddd, &eee, aaa, bbb, X[ 0], 11);
   _mm_JJJ(&bbb, ccc, &ddd, eee, aaa, X[ 9], 13);
   _mm_JJJ(&aaa, bbb, &ccc, ddd, eee, X[ 2], 15);
   _mm_JJJ(&eee, aaa, &bbb, ccc, ddd, X[11], 15);
   _mm_JJJ(&ddd, eee, &aaa, bbb, ccc, X[ 4],  5);
   _mm_JJJ(&ccc, ddd, &eee, aaa, bbb, X[13],  7);
   _mm_JJJ(&bbb, ccc, &ddd, eee, aaa, X[ 6],  7);
   _mm_JJJ(&aaa, bbb, &ccc, ddd, eee, X[15],  8);
   _mm_JJJ(&eee, aaa, &bbb, ccc, ddd, X[ 8], 11);
   _mm_JJJ(&ddd, eee, &aaa, bbb, ccc, X[ 1], 14);
   _mm_JJJ(&ccc, ddd, &eee, aaa, bbb, X[10], 14);
   _mm_JJJ(&bbb, ccc, &ddd, eee, aaa, X[ 3], 12);
   _mm_JJJ(&aaa, bbb, &ccc, ddd, eee, X[12],  6);

   /* parallel round 2 */
   _mm_III(&eee, aaa, &bbb, ccc, ddd, X[ 6],  9);
   _mm_III(&ddd, eee, &aaa, bbb, ccc, X[11], 13);
   _mm_III(&ccc, ddd, &eee, aaa, bbb, X[ 3], 15);
   _mm_III(&bbb, ccc, &ddd, eee, aaa, X[ 7],  7);
   _mm_III(&aaa, bbb, &ccc, ddd, eee, X[ 0], 12);
   _mm_III(&eee, aaa, &bbb, ccc, ddd, X[13],  8);
   _mm_III(&ddd, eee, &aaa, bbb, ccc, X[ 5],  9);
   _mm_III(&ccc, ddd, &eee, aaa, bbb, X[10], 11);
   _mm_III(&bbb, ccc, &ddd, eee, aaa, X[14],  7);
   _mm_III(&aaa, bbb, &ccc, ddd, eee, X[15],  7);
   _mm_III(&eee, aaa, &bbb, ccc, ddd, X[ 8], 12);
   _mm_III(&ddd, eee, &aaa, bbb, ccc, X[12],  7);
   _mm_III(&ccc, ddd, &eee, aaa, bbb, X[ 4],  6);
   _mm_III(&bbb, ccc, &ddd, eee, aaa, X[ 9], 15);
   _mm_III(&aaa, bbb, &ccc, ddd, eee, X[ 1], 13);
   _mm_III(&eee, aaa, &bbb, ccc, ddd, X[ 2], 11);

   /* parallel round 3 */
   _mm_HHH(&ddd, eee, &aaa, bbb, ccc, X[15],  9);
   _mm_HHH(&ccc, ddd, &eee, aaa, bbb, X[ 5],  7);
   _mm_HHH(&bbb, ccc, &ddd, eee, aaa, X[ 1], 15);
   _mm_HHH(&aaa, bbb, &ccc, ddd, eee, X[ 3], 11);
   _mm_HHH(&eee, aaa, &bbb, ccc, ddd, X[ 7],  8);
   _mm_HHH(&ddd, eee, &aaa, bbb, ccc, X[14],  6);
   _mm_HHH(&ccc, ddd, &eee, aaa, bbb, X[ 6],  6);
   _mm_HHH(&bbb, ccc, &ddd, eee, aaa, X[ 9], 14);
   _mm_HHH(&aaa, bbb, &ccc, ddd, eee, X[11], 12);
   _mm_HHH(&eee, aaa, &bbb, ccc, ddd, X[ 8], 13);
   _mm_HHH(&ddd, eee, &aaa, bbb, ccc, X[12],  5);
   _mm_HHH(&ccc, ddd, &eee, aaa, bbb, X[ 2], 14);
   _mm_HHH(&bbb, ccc, &ddd, eee, aaa, X[10], 13);
   _mm_HHH(&aaa, bbb, &ccc, ddd, eee, X[ 0], 13);
   _mm_HHH(&eee, aaa, &bbb, ccc, ddd, X[ 4],  7);
   _mm_HHH(&ddd, eee, &aaa, bbb, ccc, X[13],  5);

   /* parallel round 4 */
   _mm_GGG(&ccc, ddd, &eee, aaa, bbb, X[ 8], 15);
   _mm_GGG(&bbb, ccc, &ddd, eee, aaa, X[ 6],  5);
   _mm_GGG(&aaa, bbb, &ccc, ddd, eee, X[ 4],  8);
   _mm_GGG(&eee, aaa, &bbb, ccc, ddd, X[ 1], 11);
   _mm_GGG(&ddd, eee, &aaa, bbb, ccc, X[ 3], 14);
   _mm_GGG(&ccc, ddd, &eee, aaa, bbb, X[11], 14);
   _mm_GGG(&bbb, ccc, &ddd, eee, aaa, X[15],  6);
   _mm_GGG(&aaa, bbb, &ccc, ddd, eee, X[ 0], 14);
   _mm_GGG(&eee, aaa, &bbb, ccc, ddd, X[ 5],  6);
   _mm_GGG(&ddd, eee, &aaa, bbb, ccc, X[12],  9);
   _mm_GGG(&ccc, ddd, &eee, aaa, bbb, X[ 2], 12);
   _mm_GGG(&bbb, ccc, &ddd, eee, aaa, X[13],  9);
   _mm_GGG(&aaa, bbb, &ccc, ddd, eee, X[ 9], 12);
   _mm_GGG(&eee, aaa, &bbb, ccc, ddd, X[ 7],  5);
   _mm_GGG(&ddd, eee, &aaa, bbb, ccc, X[10], 15);
   _mm_GGG(&ccc, ddd, &eee, aaa, bbb, X[14],  8);

   /* parallel round 5 */
   _mm_FFF(&bbb, ccc, &ddd, eee, aaa, X[12] ,  8);
   _mm_FFF(&aaa, bbb, &ccc, ddd, eee, X[15] ,  5);
   _mm_FFF(&eee, aaa, &bbb, ccc, ddd, X[10] , 12);
   _mm_FFF(&ddd, eee, &aaa, bbb, ccc, X[ 4] ,  9);
   _mm_FFF(&ccc, ddd, &eee, aaa, bbb, X[ 1] , 12);
   _mm_FFF(&bbb, ccc, &ddd, eee, aaa, X[ 5] ,  5);
   _mm_FFF(&aaa, bbb, &ccc, ddd, eee, X[ 8] , 14);
   _mm_FFF(&eee, aaa, &bbb, ccc, ddd, X[ 7] ,  6);
   _mm_FFF(&ddd, eee, &aaa, bbb, ccc, X[ 6] ,  8);
   _mm_FFF(&ccc, ddd, &eee, aaa, bbb, X[ 2] , 13);
   _mm_FFF(&bbb, ccc, &ddd, eee, aaa, X[13] ,  6);
   _mm_FFF(&aaa, bbb, &ccc, ddd, eee, X[14] ,  5);
   _mm_FFF(&eee, aaa, &bbb, ccc, ddd, X[ 0] , 15);
   _mm_FFF(&ddd, eee, &aaa, bbb, ccc, X[ 3] , 13);
   _mm_FFF(&ccc, ddd, &eee, aaa, bbb, X[ 9] , 11);
   _mm_FFF(&bbb, ccc, &ddd, eee, aaa, X[11] , 11);

   /* combine results */
   ddd = _mm_add_epi32(ddd,_mm_add_epi32(cc,MDbuf[1]));
   MDbuf[1] = _mm_add_epi32(MDbuf[2],_mm_add_epi32(dd,eee));
   MDbuf[2] = _mm_add_epi32(MDbuf[3],_mm_add_epi32(ee,aaa));
   MDbuf[3] = _mm_add_epi32(MDbuf[4],_mm_add_epi32(aa,bbb));
   MDbuf[4] = _mm_add_epi32(MDbuf[0],_mm_add_epi32(bb,ccc));
   MDbuf[0] = ddd;

//   return;
}


/********************************************************************/

void MM_MDfinish(__m128i *MDbuf, __m128i *strptr, dword lswlen, dword mswlen)
{
   unsigned int i;                                 /* counter       */
   uint32_t        X[16 * 4];                      /* message words */
   __m128i *XPrt = (__m128i*) X;
//	const __m128i vm = _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
//	const __m128i vm = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

//   memset(X, 0, 16*4*sizeof(dword));
   for (i=0; i<((lswlen/4)&15); i++) {
//	   XPrt[i] = _mm_shuffle_epi8(strptr[i],vm);
	   XPrt[i] = strptr[i];
   }
   for (;i<16;i++)
	   XPrt[i] = _mm_setzero_si128();
//   /* put bytes from strptr into X */
//   for (i=0; i<(lswlen&63); i++) {
//      /* byte i goes into word X[i div 4] at pos.  8*(i mod 4)  */
//      X[i>>2] ^= (dword) *strptr++ << (8 * (i&3));
//   }

   /* append the bit m_n == 1 */
//   X[(lswlen>>2)&15] ^= (dword)1 << (8*(lswlen&3) + 7);
   XPrt[(lswlen>>2)&15] ^= _mm_set1_epi32 ((uint32_t) 1<<(8*(lswlen&3) + 7));

//   this section still needs to be converted/added for real rnd160--> GDR
//   but since the size is known here, this step is not performed.
//   if ((lswlen & 63) > 55) {
//      /* length goes to next block */
//      compress(MDbuf, X);
//      memset(X, 0, 16*sizeof(dword));
//   }

   /* append length in bits*/
   XPrt[14] ^= _mm_set1_epi32 (lswlen<<3);
   XPrt[15] ^= _mm_set1_epi32 ((lswlen >> 29) | (mswlen << 3));

//   X[14] = lswlen << 3;
//  X[15] = (lswlen >> 29) | (mswlen << 3);
   MM_compress(MDbuf, XPrt);

//   return;
}


void MDfinish(dword *MDbuf, byte *strptr, dword lswlen, dword mswlen)
{
   unsigned int i;                                 /* counter       */
   dword        X[16];                             /* message words */

   memset(X, 0, 16*sizeof(dword));

   /* put bytes from strptr into X */
   for (i=0; i<(lswlen&63); i++) {
      /* byte i goes into word X[i div 4] at pos.  8*(i mod 4)  */
      X[i>>2] ^= (dword) *strptr++ << (8 * (i&3));
   }

   /* append the bit m_n == 1 */
   X[(lswlen>>2)&15] ^= (dword)1 << (8*(lswlen&3) + 7);

   if ((lswlen & 63) > 55) {
      /* length goes to next block */
      compress(MDbuf, X);
      memset(X, 0, 16*sizeof(dword));
   }

   /* append length in bits*/
   X[14] = lswlen << 3;
   X[15] = (lswlen >> 29) | (mswlen << 3);
   compress(MDbuf, X);

//   return;
}

void MM_matrix_transpose_r2c(__m128i* inbuf,__m128i* outbuf, uint32_t rows, uint32_t colums)
{
	// this matrix transpose changes rows in to colums using block modes
	__m128i block[4];
	uint32_t col = colums/4;
	uint32_t i; // index for the transformation
//	__m128i *inbufPtr = (__m128i*) inbuf;
//	__m128i *outbufPtr = (__m128i*) outbuf;

	for (i=0; i<col;i++){
		block[0] = _mm_unpacklo_epi32(inbuf[i], inbuf[i + 2*col]);
		block[1] = _mm_unpacklo_epi32(inbuf[i + col], inbuf[i + 3*col]);
		block[2] = _mm_unpackhi_epi32(inbuf[i], inbuf[i + 2*col]);
		block[3] = _mm_unpackhi_epi32(inbuf[i+ col], inbuf[i + 3*col]);
		outbuf[i*rows+0] = _mm_unpacklo_epi32(block[0], block[1]);
		outbuf[i*rows+1] = _mm_unpackhi_epi32(block[0], block[1]);
		outbuf[i*rows+2] = _mm_unpacklo_epi32(block[2], block[3]);
		outbuf[i*rows+3] = _mm_unpackhi_epi32(block[2], block[3]);
	}

}

void MM_matrix_transpose_c2r(__m128i* inbuf,__m128i* outbuf, uint32_t rows, uint32_t colums)
{
	// this matrix transpose changes rows in to colums using block modes
	__m128i block[4];
	uint32_t row = rows/4;
	uint32_t i; // index for the transformation
//	__m128i *inbufPtr = (__m128i*) inbuf;
//	__m128i *outbufPtr = (__m128i*) outbuf;
	for (i=0; i<row;i++){
		block[0] = _mm_unpacklo_epi32(inbuf[i*colums], inbuf[i*colums + 2]);
		block[1] = _mm_unpacklo_epi32(inbuf[i*colums + 1], inbuf[i*colums + 3]);
		block[2] = _mm_unpackhi_epi32(inbuf[i*colums], inbuf[i*colums + 2]);
		block[3] = _mm_unpackhi_epi32(inbuf[i*colums+ 1], inbuf[i*colums + 3]);
		outbuf[i] = _mm_unpacklo_epi32(block[0], block[1]);
		outbuf[i+row] = _mm_unpackhi_epi32(block[0], block[1]);
		outbuf[i+row*2] = _mm_unpacklo_epi32(block[2], block[3]);
		outbuf[i+row*3] = _mm_unpackhi_epi32(block[2], block[3]);
	}

}


/************************ end of file rmd160.c **********************/

