/*
 * Copyright 2014 Kangaderoo
 */

#include <string.h>
#include <immintrin.h>
#include <inttypes.h>
#include "sha256.h"

// #define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
// #define Maj(x, y, z)    ((x & (y | z)) | (y & z))

__m128i funct_S0_Maj(__m128i *x, __m128i *y, __m128i *z)
{
        __m128i _calc_maj1 = _mm_or_si128(*y,*z);
        __m128i _calc_maj2 = _mm_and_si128(*y,*z);
        __m128i rot2 = _mm_srli_epi32(*x, 2);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 2));
        __m128i rot13_22 = _mm_srli_epi32(*x, 13);

        rot2 = _mm_xor_si128(rot2, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 13));
        rot2 = _mm_xor_si128(rot2, rot13_22);
        rot13_22 = _mm_srli_epi32(*x, 22);
        rot2 = _mm_xor_si128(rot2, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 22));
        rot2 = _mm_xor_si128(rot2, rot13_22);
        _calc_maj1 = _mm_and_si128(_calc_maj1, *x);
        rot2 = _mm_xor_si128(rot2, _calc);
        _calc_maj1 = _mm_or_si128(_calc_maj1, _calc_maj2);
        _calc = _mm_add_epi32(_calc_maj1, rot2);
        return _calc;
}


__m128i funct_S0(const __m128i *x)
{
        __m128i rot2 = _mm_srli_epi32(*x, 2);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 2));
        __m128i rot13_22 = _mm_srli_epi32(*x, 13);

        rot2 = _mm_xor_si128(rot2, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 13));
        rot2 = _mm_xor_si128(rot2, rot13_22);
        rot13_22 = _mm_srli_epi32(*x, 22);
        rot2 = _mm_xor_si128(rot2, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 22));
        rot2 = _mm_xor_si128(rot2, rot13_22);
        _calc = _mm_xor_si128(_calc, rot2);
        return _calc;
}

__m128i funct_S0_old(const __m128i *x)
{
        __m128i rot2 = _mm_srli_epi32(*x, 2);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 2));

        __m128i rot13 = _mm_srli_epi32(*x, 13);
        rot13 = _mm_xor_si128(rot13, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 13));

        __m128i rot22 = _mm_srli_epi32(*x, 22);
        rot22 = _mm_xor_si128(rot22, _calc);

        _calc = _mm_slli_epi32(*x,(32 - 22));
        _calc = _mm_xor_si128(rot22, _calc);
        _calc = _mm_xor_si128(_calc, rot2);
        _calc = _mm_xor_si128(_calc, rot13);

        return _calc;
}

// #define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
// #define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
__m128i funct_S1_Ch(__m128i *x, __m128i *y, __m128i *z)
{
        __m128i rot6 = _mm_srli_epi32(*x, 6);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 6));
        __m128i rot11_25 = _mm_srli_epi32(*x, 11);
    	__m128i calc_Ch = _mm_xor_si128(*y,*z);

        rot6 = _mm_xor_si128(rot6, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 11));
        rot6 = _mm_xor_si128(rot6, rot11_25);
        rot11_25 = _mm_srli_epi32(*x, 25);
        rot6 = _mm_xor_si128(rot6, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 25));
        rot6 = _mm_xor_si128(rot6, rot11_25);
        calc_Ch = _mm_and_si128(calc_Ch, *x);
        rot6 = _mm_xor_si128(rot6, _calc);
        calc_Ch = _mm_xor_si128(calc_Ch, *z);
        _calc = _mm_add_epi32(rot6, calc_Ch);
        return _calc;
}
// #define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
__m128i funct_S1(const __m128i *x)
{
        __m128i rot6 = _mm_srli_epi32(*x, 6);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 6));
        __m128i rot11_25 = _mm_srli_epi32(*x, 11);

        rot6 = _mm_xor_si128(rot6, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 11));
        rot6 = _mm_xor_si128(rot6, rot11_25);
        rot11_25 = _mm_srli_epi32(*x, 25);
        rot6 = _mm_xor_si128(rot6, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 25));
        rot6 = _mm_xor_si128(rot6, rot11_25);
        _calc = _mm_xor_si128(_calc, rot6);

        return _calc;
}

__m128i funct_S1_old(const __m128i *x)
{
        __m128i rot6 = _mm_srli_epi32(*x, 6);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 6));

        __m128i rot11 = _mm_srli_epi32(*x, 11);
        rot11 = _mm_xor_si128(rot11, _calc);
        _calc = _mm_slli_epi32(*x,(32 - 11));

        __m128i rot25 = _mm_srli_epi32(*x, 25);
        rot25 = _mm_xor_si128(rot25, _calc);

        _calc = _mm_slli_epi32(*x,(32 - 25));
        _calc = _mm_xor_si128(rot25, _calc);
        _calc = _mm_xor_si128(_calc, rot6);
        _calc = _mm_xor_si128(_calc, rot11);

        return _calc;
}
// #define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
__m128i funct_s0(const __m128i *x)
{
        __m128i rot7 = _mm_srli_epi32(*x, 7);
        __m128i rot18 = _mm_srli_epi32(*x, 18);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 7));
        __m128i shift3 = _mm_srli_epi32(*x, 3);

        _calc = _mm_xor_si128(_calc, rot7);
        _calc = _mm_xor_si128(_calc, rot18);

        rot18 = _mm_slli_epi32(*x,(32 - 18));
        _calc = _mm_xor_si128(_calc, rot18);

        _calc = _mm_xor_si128(_calc, shift3);

        return _calc;
}

// #define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))
__m128i funct_s1(const __m128i *x)
{
        __m128i rot17 = _mm_srli_epi32(*x, 17);
        __m128i rot19 = _mm_srli_epi32(*x, 19);
        __m128i _calc = _mm_slli_epi32(*x,(32 - 17));
        __m128i shift10 = _mm_srli_epi32(*x, 10);

        _calc = _mm_xor_si128(_calc, rot17);
        _calc = _mm_xor_si128(_calc, rot19);

        rot19 =  _mm_slli_epi32(*x,(32 - 19));
        _calc = _mm_xor_si128(_calc, rot19);

        _calc = _mm_xor_si128(_calc, shift10);

        return _calc;
}

// #define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
__m128i funct_Ch(__m128i *x, __m128i *y, __m128i *z)
{
        __m128i _calc = _mm_xor_si128(*y,*z);
        _calc = _mm_and_si128(_calc, *x);
        _calc = _mm_xor_si128(_calc, *z);
        return _calc;
}

// #define Maj(x, y, z)    ((x & (y | z)) | (y & z))
__m128i funct_Maj(__m128i *x, __m128i *y, __m128i *z)
{
        __m128i _calc = _mm_or_si128(*y,*z);
        __m128i _calc2 = _mm_and_si128(*y,*z);
        _calc = _mm_and_si128(_calc, *x);
        _calc = _mm_or_si128(_calc, _calc2);
        return _calc;
}

/* Elementary functions used by SHA256 */
//#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
//#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
//#define ROTR(x, n)      ((x >> n) | (x << (32 - n)))
//#define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
//#define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
//#define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
//#define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

/* SHA256 round function */
/*
 *  #define RND(a, b, c, d, e, f, g, h, k) \
 *	do { \
 *		t0 = h + S1(e) + Ch(e, f, g) + k; \
 *		t1 = S0(a) + Maj(a, b, c); \
 *		d += t0; \
 *		h  = t0 + t1; \
 *	} while (0)
 */
void MM_RND(__m128i *a, __m128i *b, __m128i *c, __m128i *d, __m128i *e, __m128i *f, __m128i *g, __m128i *h,
		          const __m128i k)
{
	__m128i t0;
	__m128i t1;
	__m128i _calc;

	t0 = _mm_add_epi32(k, *h);
	_calc = funct_S1_Ch(e,f,g);
	t1 = funct_S0_Maj(a,b,c);
	t0 = _mm_add_epi32(t0, _calc);

	*d = _mm_add_epi32(*d, t0);
	*h = _mm_add_epi32(t0, t1);
}

void MM_RND_old(__m128i *a, __m128i *b, __m128i *c, __m128i *d, __m128i *e, __m128i *f, __m128i *g, __m128i *h,
		          const __m128i k)
{
	__m128i t0;
	__m128i t1;
	__m128i _calc = funct_S1(e);
	__m128i _calc1 = funct_Ch(e, f, g);
	_calc = _mm_add_epi32(_calc,_calc1);
	_calc = _mm_add_epi32(_calc,*h);
	t0 = _mm_add_epi32(_calc,k);
	_calc = funct_Maj(a,b,c);
	_calc1 = funct_S0(a);
	t1 = _mm_add_epi32(_calc,_calc1);
	*d = _mm_add_epi32(*d, t0);
	*h = _mm_add_epi32(t1, t0);
}
/* Adjusted round function for rotating state */
/*
 *  #define RNDr(S, W, i) \
 *	RND(S[(64 - i) % 8], S[(65 - i) % 8], \
 *	    S[(66 - i) % 8], S[(67 - i) % 8], \
 *	    S[(68 - i) % 8], S[(69 - i) % 8], \
 *	    S[(70 - i) % 8], S[(71 - i) % 8], \
 *	    W[i] + sha256_k_sidm[i])
 *
 */

void sha256_init(uint32_t *state)
{
	memcpy(state, sha256_h, 32);
}

void MM_sha256_init(uint32_t *state)
{
	__m128i *ConstPrt = (__m128i*) sha256_h_quad;
	__m128i *BufPrt = (__m128i*) state;
	uint32_t i;
	for(i=0;i<8;i++)
		BufPrt[i] = ConstPrt[i];
//	memcpy(state, sha256_h_quad, 32*4);
}

void MM_clear_mem(__m128i *memloc, uint32_t size)
{
	uint32_t i;
	for (i=0;i<size;i++)
		memloc[i] = _mm_setzero_si128();
}

/*
 * SHA256 block compression function.
 */

void sha256_transform(uint32_t *state,  const uint32_t *block)
{
	uint32_t W[64];
	uint32_t S[8];
	uint32_t t0, t1;
	int i;

	memcpy(W, block, 64);

	for (i = 16; i < 64; i += 2) {
		W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}

	/* 2. Initialize working variables. */
	memcpy(S, state, 32);

	/* 3. Mix. */
	RNDr(S, W,  0);
	RNDr(S, W,  1);
	RNDr(S, W,  2);
	RNDr(S, W,  3);
	RNDr(S, W,  4);
	RNDr(S, W,  5);
	RNDr(S, W,  6);
	RNDr(S, W,  7);
	RNDr(S, W,  8);
	RNDr(S, W,  9);
	RNDr(S, W, 10);
	RNDr(S, W, 11);
	RNDr(S, W, 12);
	RNDr(S, W, 13);
	RNDr(S, W, 14);
	RNDr(S, W, 15);
	RNDr(S, W, 16);
	RNDr(S, W, 17);
	RNDr(S, W, 18);
	RNDr(S, W, 19);
	RNDr(S, W, 20);
	RNDr(S, W, 21);
	RNDr(S, W, 22);
	RNDr(S, W, 23);
	RNDr(S, W, 24);
	RNDr(S, W, 25);
	RNDr(S, W, 26);
	RNDr(S, W, 27);
	RNDr(S, W, 28);
	RNDr(S, W, 29);
	RNDr(S, W, 30);
	RNDr(S, W, 31);
	RNDr(S, W, 32);
	RNDr(S, W, 33);
	RNDr(S, W, 34);
	RNDr(S, W, 35);
	RNDr(S, W, 36);
	RNDr(S, W, 37);
	RNDr(S, W, 38);
	RNDr(S, W, 39);
	RNDr(S, W, 40);
	RNDr(S, W, 41);
	RNDr(S, W, 42);
	RNDr(S, W, 43);
	RNDr(S, W, 44);
	RNDr(S, W, 45);
	RNDr(S, W, 46);
	RNDr(S, W, 47);
	RNDr(S, W, 48);
	RNDr(S, W, 49);
	RNDr(S, W, 50);
	RNDr(S, W, 51);
	RNDr(S, W, 52);
	RNDr(S, W, 53);
	RNDr(S, W, 54);
	RNDr(S, W, 55);
	RNDr(S, W, 56);
	RNDr(S, W, 57);
	RNDr(S, W, 58);
	RNDr(S, W, 59);
	RNDr(S, W, 60);
	RNDr(S, W, 61);
	RNDr(S, W, 62);
	RNDr(S, W, 63);

	/* 4. Mix local working variables into global state */
	for (i = 0; i < 8; i++)
		state[i] += S[i];
}

void MM_sha256_transform(__m128i *state,  const __m128i *block)
{

	uint32_t W[64*4] __attribute__((aligned(16)));
	uint32_t S[8*4] __attribute__((aligned(16)));

	int i;

	__m128i *WPrt = (__m128i*) W;
	__m128i *SPrt = (__m128i*) S;
	__m128i *ConstPrt = (__m128i*) sha256_k_quad;
	__m128i _calc;

	for(i=0;i<16;i++)
		WPrt[i] = block[i]; // memcpy(W, block, 64*4);

	/* 1. Prepare message schedule W. */

	for (i = 16; i < 64; i += 2) {
		// _calc = WPrt[i-2];
		WPrt[i] = funct_s1(WPrt + i - 2);
		// _calc = WPrt[i-15];
		_calc = funct_s0(WPrt + i - 15);
		WPrt[i] = _mm_add_epi32(WPrt[i], WPrt[i-7]);
		WPrt[i] = _mm_add_epi32(WPrt[i], _calc);
		WPrt[i] = _mm_add_epi32(WPrt[i], WPrt[i-16]);

		// _calc = WPrt[i-1];
		WPrt[i+1] = funct_s1(WPrt + i - 1);
		// _calc = WPrt[i-14];
		_calc = funct_s0(WPrt + i - 14);
		WPrt[i+1] = _mm_add_epi32(WPrt[i+1], WPrt[i-6]);
		WPrt[i+1] = _mm_add_epi32(WPrt[i+1], _calc);
		WPrt[i+1] = _mm_add_epi32(WPrt[i+1], WPrt[i-15]);
	}

	/* 2. Initialize working variables. */
	for(i=0;i<8;i++)
		SPrt[i] = state[i]; //	memcpy(&S, state, 32*4);

	_calc = _mm_add_epi32(WPrt[0], ConstPrt[0]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[1], ConstPrt[1]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[2], ConstPrt[2]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[3], ConstPrt[3]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[4], ConstPrt[4]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[5], ConstPrt[5]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[6], ConstPrt[6]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[7], ConstPrt[7]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	_calc = _mm_add_epi32(WPrt[8], ConstPrt[8]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[9], ConstPrt[9]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[10], ConstPrt[10]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[11], ConstPrt[11]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[12], ConstPrt[12]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[13], ConstPrt[13]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[14], ConstPrt[14]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[15], ConstPrt[15]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	_calc = _mm_add_epi32(WPrt[16], ConstPrt[16]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[17], ConstPrt[17]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[18], ConstPrt[18]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[19], ConstPrt[19]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[20], ConstPrt[20]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[21], ConstPrt[21]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[22], ConstPrt[22]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[23], ConstPrt[23]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	_calc = _mm_add_epi32(WPrt[24], ConstPrt[24]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[25], ConstPrt[25]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[26], ConstPrt[26]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[27], ConstPrt[27]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[28], ConstPrt[28]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[29], ConstPrt[29]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[30], ConstPrt[30]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[31], ConstPrt[31]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	_calc = _mm_add_epi32(WPrt[32], ConstPrt[32]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[33], ConstPrt[33]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[34], ConstPrt[34]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[35], ConstPrt[35]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[36], ConstPrt[36]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[37], ConstPrt[37]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[38], ConstPrt[38]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[39], ConstPrt[39]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	_calc = _mm_add_epi32(WPrt[40], ConstPrt[40]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[41], ConstPrt[41]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[42], ConstPrt[42]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[43], ConstPrt[43]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[44], ConstPrt[44]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[45], ConstPrt[45]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[46], ConstPrt[46]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[47], ConstPrt[47]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);


	_calc = _mm_add_epi32(WPrt[48], ConstPrt[48]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[49], ConstPrt[49]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[50], ConstPrt[50]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[51], ConstPrt[51]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[52], ConstPrt[52]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[53], ConstPrt[53]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[54], ConstPrt[54]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[55], ConstPrt[55]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	_calc = _mm_add_epi32(WPrt[56], ConstPrt[56]);
	MM_RND(SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7, _calc);
	_calc = _mm_add_epi32(WPrt[57], ConstPrt[57]);
	MM_RND(SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6, _calc);
	_calc = _mm_add_epi32(WPrt[58], ConstPrt[58]);
	MM_RND(SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5, _calc);
	_calc = _mm_add_epi32(WPrt[59], ConstPrt[59]);
	MM_RND(SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3,SPrt+4, _calc);
	_calc = _mm_add_epi32(WPrt[60], ConstPrt[60]);
	MM_RND(SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2,SPrt+3, _calc);
	_calc = _mm_add_epi32(WPrt[61], ConstPrt[61]);
	MM_RND(SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1,SPrt+2, _calc);
	_calc = _mm_add_epi32(WPrt[62], ConstPrt[62]);
	MM_RND(SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0,SPrt+1, _calc);
	_calc = _mm_add_epi32(WPrt[63], ConstPrt[63]);
	MM_RND(SPrt+1,SPrt+2,SPrt+3,SPrt+4,SPrt+5,SPrt+6,SPrt+7,SPrt+0, _calc);

	/* 4. Mix local working variables into global state */
	for (i = 0; i < 8; i++)
		state[i] = _mm_add_epi32(state[i],SPrt[i]);
}

void MM_beRecode(__m128i *block, uint32_t length)
{
	uint32_t i;
	const __m128i vm = _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
	for (i=0;i<length;i++)
		block[i] = _mm_shuffle_epi8(block[i],vm);
}


void sha256_finish(uint32_t *state,  const unsigned char *block,  int blcklen, int swap)
{
	   unsigned int    i;        			              /* counters      */
	   int   		   j;
	   uint32_t        S[16];                             /* message words */

		// make chunk blocks of 512 bits.
		// if the length exceeds the 512 bits call the transform function

		// append 1 bit of "1" to the message

	   j = 0;
	   do{
		   memset(S, 0, 16*sizeof(uint32_t));
		   // empty string check
		   // S[0]=0x80000000;
		   // sha256_transform(state,S);
		   // 0x e3b0c442 98fc1c14 9afbf4c8 996fb924 27ae41e4 649b934c a495991b 7852b855
		   if (j<blcklen-64){
			   /* put bytes from block into S 512 bits */
			   for (i=0; i<64; i++) {
				  /* byte i goes into word X[i div 4] at pos.  8*(i mod 4)  */
				  S[i>>2] ^= (uint32_t) *block++ << (8 * ((3-i)&3));
			   }
			   sha256_transform(state,S);
		   }else{
			   if (j<blcklen+9){// is there enough room to add the 1 and length info?
				   for (i=0; i<blcklen-j; i++) {
					  /* byte i goes into word X[i div 4] at pos.  8*(i mod 4)  */
					  S[i>>2] ^= (uint32_t) *block++ << (8 * ((3-i)&3));
				   }
				   // add the "1" and the length
 				   S[i>>2] ^= (uint32_t) 1 << (8 * ((3-i)&3) + 7);
 				   S[14] = (blcklen >> 29);
 				   S[15] = blcklen << 3;
			   }else{
				   for (i=0; i<blcklen-j; i++) {
					  /* byte i goes into word X[i div 4] at pos.  8*(i mod 4)  */
					  S[i>>2] ^= (uint32_t) *block++ << (8 * ((3-i)&3));
				   }
 				   S[i>>2] ^= (uint32_t) 1 << (8 * (i&3) + 7);
				   sha256_transform(state,S);
				   // move the length info into the next block
				   memset(S, 0, 16*sizeof(uint32_t));
 				   S[14] = (blcklen >> 29);
 				   S[15] = blcklen << 3;
			   }
		   }
		   j = j + 64;
	   }while (j<blcklen);

	   sha256_transform(state,S);

	   return;
}
