/*
 * custom_ec_bn.h
 *
 *  Created on: Jul 18, 2014
 *      Author: kangaderoo
 *
 *      this file containes EC and BN operations that use minimized instructions
 *      specific for the bitcoin EC curve.
 *      The BN operations use SIDM instructions to perform add sub and multiply instructions in less clock pulses
 *      the field modulation function is optimized for a field size close to a mersenne prime (2^n minus a small number)
 *
 */



#ifndef CUSTOM_EC_BN_H_
#define CUSTOM_EC_BN_H_

#include <openssl/ec.h>
#include <openssl/bn.h>

#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>


/*
 * This structure is to calculate with big numbers using the SIDM
 * a 256 bit will be moved to three 128 bit values.
 * it will be configured as this <MSBbits->32><carry1>56bit<carry2>56bit<carry3>56bit<carry4>56bit.
 * multiply will be done with add and shift.
 * the modulo will be done after each add/shift/sub operation.
 */
/*this is the modulo field used for bitcoin*/
static const uint32_t pfield[12] __attribute__((aligned(16))) =
			{0xffffffff,0x0,0x0,0x0,0xffffffff,0xffffff,0xffffffff,0xffffff,
			 0xfffffc2f,0xfffffe,0xffffffff,0xffffff} ;
/* this is 2n - pfield */
static const uint32_t pfld_rem[12] __attribute__((aligned(16))) =
			{0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
			 0x3d1,0x1,0x0,0x0} ;

//const __m128i clr_hig_carry = _mm_setr_epi64(0x0100000000000000, 0x0000000000000000);
//const __m128i clr_low_carry = _mm_setr_epi64(0x0000000000000000, 0x0100000000000000);


struct _sidm_bn_context_s{
	uint32_t BNbuffer[12] __attribute__((aligned(16)));
};

typedef struct _sidm_bn_context_s _sidm_bn_context_t;

/*
 * move a 256bit BN to the sidm context struct for calculations
 */
void BN_to_Struct(const BIGNUM *BN256Bits, _sidm_bn_context_t *sidm_calc_context);

/*
 * move a  sidm context struct for calculations to a BigNum
 */
void Struct_to_BN(_sidm_bn_context_t *sidm_calc_context, BIGNUM *BN256Bits);


void struct_BN_add(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a, const _sidm_bn_context_t *calc_context_b);

void struct_BN_shl(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a);

void struct_BN_sub(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a, const _sidm_bn_context_t *calc_context_b);

void struct_BN_mul(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a, const _sidm_bn_context_t *calc_context_b);

/*
 * Add two affine EC points and return the Jacobian coordinates
 */
void struct_BN_EC_Point_Add_Affine(const EC_GROUP *group, EC_POINT *r, const EC_POINT *a, const EC_POINT *b, BN_CTX *ctx);

size_t struct_EC_POINT_point2oct(const EC_GROUP *group, const EC_POINT *p,	point_conversion_form_t form,
        unsigned char *buf, size_t len, BN_CTX *ctx);

void BN_EC_Point_Add_Affine(const EC_GROUP *group, EC_POINT *r, const EC_POINT *a, const EC_POINT *b, BN_CTX *ctx);


#endif /* CUSTOM_EC_BN_H_ */
