/*
 * custom_ec_bn.c
 *
 *  Created on: Jul 18, 2014
 *      Author: kangaderoo
 */

#include <openssl/bn.h>
#include <openssl/ec.h>
#include <immintrin.h>
#include <string.h>
#include <inttypes.h>

#include "custom_ec_bn.h"


void swap_64bit_endian(_sidm_bn_context_t *sidm_calc_context)
{
	__m128i *content = (__m128i*) sidm_calc_context->BNbuffer;
	const __m128i vm = _mm_setr_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
	content[0] = _mm_shuffle_epi8(content[0],vm);
	content[1] = _mm_shuffle_epi8(content[1],vm);
	content[2] = _mm_shuffle_epi8(content[2],vm);
}


void BN_to_Struct(const BIGNUM *BN256Bits, _sidm_bn_context_t *sidm_calc_context)
{
	unsigned char *context;
	__m128i *content = (__m128i*) sidm_calc_context->BNbuffer;
	uint32_t i, step, len;
	context = (unsigned char*) sidm_calc_context->BNbuffer;
	len = BN_num_bytes(BN256Bits);
	content[0] = _mm_setzero_si128();
	content[1] = _mm_setzero_si128();
	content[2] = _mm_setzero_si128();
	len = BN_bn2bin(BN256Bits, context+(48-len));
	step = 4;
	for (i=12;i<41;i++){
		if ((i % 8) == 0){
			context[i] = 0;
			step--;
		}else{
			context[i]= context[i+step];
		}
	}
	swap_64bit_endian(sidm_calc_context);
}

void Struct_to_BN(_sidm_bn_context_t *sidm_calc_context, BIGNUM *BN256Bits)
{
	unsigned char *context;
	uint32_t i, step;
	context = (unsigned char*) sidm_calc_context->BNbuffer;
	swap_64bit_endian(sidm_calc_context);
	step = 0;
	for (i=40;i>15;i--){
		if ((i % 8) == step){
			step++;
		}
		context[i]= context[i-step];
	}
	BN_bin2bn(context+16, 32 ,BN256Bits);
	/*
	 * there is still a very slim chance that the result is 256 bits but bigger than the field size
	 * when the BN is inserted into a EC_Point the EC import will handle this, a check would be nice though.
	 */
}


void struct_BN_add(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a, const _sidm_bn_context_t *calc_context_b)
{
 	unsigned char *context;
	__m128i *res = (__m128i*) calc_context_r->BNbuffer;
	__m128i *var_a = (__m128i*) calc_context_a->BNbuffer;
	__m128i *var_b = (__m128i*) calc_context_b->BNbuffer;
	__m128i *remfield = (__m128i*) &pfld_rem;
	const __m128i hig_carry = _mm_setr_epi8(0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0);
	const __m128i low_carry = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

	context = (unsigned char*) calc_context_r->BNbuffer;

	res[0] = _mm_add_epi64(var_a[0], var_b[0]);
	res[1] = _mm_add_epi64(var_a[1], var_b[1]);
	res[2] = _mm_add_epi64(var_a[2], var_b[2]);
	/*
	 * now check and handle the carry's
	 */
	if (context[39]){
		res[2] = _mm_add_epi64(res[2], hig_carry);
		context[39] = 0;
	}
	if (context[47]){
		res[1] = _mm_add_epi64(res[1], low_carry);
		context[47] = 0;
	}
	if (context[23]){
		res[1] = _mm_add_epi64(res[1], hig_carry);
		context[23] = 0;
	}
	if (context[31]){
		res[0] = _mm_add_epi64(res[0], low_carry);
		context[31] = 0;
	}
	// check if the number exceeds 256 bits
	// if so add the (2n - modulo field variable) and remove the carry bit.
	if (context[12]){
		res[2] = _mm_add_epi64(res[2], remfield[2]);
		/*
		 * now check and handle the carry's again
		 */
		if (context[39]){
			res[2] = _mm_add_epi64(res[2], hig_carry);
			context[39] = 0;
		}
		if (context[47]){
			res[1] = _mm_add_epi64(res[1], low_carry);
			context[47] = 0;
		}
		if (context[23]){
			res[1] = _mm_add_epi64(res[1], hig_carry);
			context[23] = 0;
		}
		if (context[31]){
			res[0] = _mm_add_epi64(res[0], low_carry);
			context[31] = 0;
		}
		context[12] = 0;
	}
}

void struct_BN_shl(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a)
{
 	unsigned char *context;
	__m128i *res = (__m128i*) calc_context_r->BNbuffer;
	__m128i *var_a = (__m128i*) calc_context_a->BNbuffer;
	__m128i *remfield = (__m128i*) &pfld_rem;
	const __m128i hig_carry = _mm_setr_epi8(0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0);
	const __m128i low_carry = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

	context = (unsigned char*) calc_context_r->BNbuffer;

	res[0] = _mm_slli_epi64(var_a[0],0x01);
	res[1] = _mm_slli_epi64(var_a[1],0x01);
	res[2] = _mm_slli_epi64(var_a[2],0x01);
	/*
	 * now check and handle the carry's
	 */
	if (context[39]){
		res[2] = _mm_add_epi64(res[2], hig_carry);
		context[39] = 0;
	}
	if (context[47]){
		res[1] = _mm_add_epi64(res[1], low_carry);
		context[47] = 0;
	}
	if (context[23]){
		res[1] = _mm_add_epi64(res[1], hig_carry);
		context[23] = 0;
	}
	if (context[31]){
		res[0] = _mm_add_epi64(res[0], low_carry);
		context[31] = 0;
	}
	// check if the number exceeds 256 bits
	// if so add the (2n - modulo field variable) and remove the carry bit.
	if (context[12]){
		res[2] = _mm_add_epi64(res[2], remfield[2]);
		/*
		 * now check and handle the carry's again
		 */
		if (context[39]){
			res[2] = _mm_add_epi64(res[2], hig_carry);
			context[39] = 0;
		}
		if (context[47]){
			res[1] = _mm_add_epi64(res[1], low_carry);
			context[47] = 0;
		}
		if (context[23]){
			res[1] = _mm_add_epi64(res[1], hig_carry);
			context[23] = 0;
		}
		if (context[31]){
			res[0] = _mm_add_epi64(res[0], low_carry);
			context[31] = 0;
		}
		context[12] = 0;
	}
}

void struct_BN_sub(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a, const _sidm_bn_context_t *calc_context_b)
{
 	unsigned char *context;
	__m128i *res = (__m128i*) calc_context_r->BNbuffer;
	__m128i *var_a = (__m128i*) calc_context_a->BNbuffer;
	__m128i *var_b = (__m128i*) calc_context_b->BNbuffer;
	__m128i *addfield = (__m128i*) &pfield;
	const __m128i hig_carry = _mm_setr_epi8(0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0);
	const __m128i low_carry = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

	context = (unsigned char*) calc_context_r->BNbuffer;

	res[0] = _mm_sub_epi64(var_a[0], var_b[0]);
	res[1] = _mm_sub_epi64(var_a[1], var_b[1]);
	res[2] = _mm_sub_epi64(var_a[2], var_b[2]);
	/*
	 * now check and handle the carry's
	 */
	if (context[39]){
		res[2] = _mm_sub_epi64(res[2], hig_carry);
		context[39] = 0;
	}
	if (context[47]){
		res[1] = _mm_sub_epi64(res[1], low_carry);
		context[47] = 0;
	}
	if (context[23]){
		res[1] = _mm_sub_epi64(res[1], hig_carry);
		context[23] = 0;
	}
	if (context[31]){
		res[0] = _mm_sub_epi64(res[0], low_carry);
		context[31] = 0;
	}
	//check if the number itself is negative
	//if yes, add the modulo field variable.
	if (context[12]){
		res[0] = _mm_sub_epi64(res[0], addfield[0]);
		res[1] = _mm_sub_epi64(res[1], addfield[1]);
		res[2] = _mm_sub_epi64(res[2], addfield[2]);
		/*
		 * now check and handle the carry's
		 */
		if (context[39]){
			res[2] = _mm_sub_epi64(res[2], hig_carry);
			context[39] = 0;
		}
		if (context[47]){
			res[1] = _mm_sub_epi64(res[1], low_carry);
			context[47] = 0;
		}
		if (context[23]){
			res[1] = _mm_sub_epi64(res[1], hig_carry);
			context[23] = 0;
		}
		if (context[31]){
			res[0] = _mm_sub_epi64(res[0], low_carry);
			context[31] = 0;
		}
		// the result should always fit the field size going from neg to pos, nothing needs to be cleared.
	}

}

/* can be used as r = a*b or a' = a*b */
/* it might speed up if first is checked if a or b is less bits, and loop over the smallest variable */

void struct_BN_mul(_sidm_bn_context_t *calc_context_r, _sidm_bn_context_t *calc_context_a, const _sidm_bn_context_t *calc_context_b)
{
 	unsigned char *context;
	context = (unsigned char*) calc_context_b->BNbuffer;
	__m128i *res = (__m128i*) calc_context_r->BNbuffer;
	__m128i *var_a = (__m128i*) calc_context_a->BNbuffer;
	__m128i *var_b = (__m128i*) calc_context_b->BNbuffer;
	_sidm_bn_context_t buffer;
	__m128i *content = (__m128i*) &buffer.BNbuffer;

	content[0] = _mm_setzero_si128();
	content[1] = _mm_setzero_si128();
	content[2] = _mm_setzero_si128();
	/*
	 * move through the 256 bits of var_b msb to lsb. if "1" then add, else shift left.
	 */
	uint32_t msb_order[32] = {11,10,9,8,
							  22,21,20,19,18,17,16,
							  30,29,28,27,26,25,24,
							  38,37,36,35,34,33,32,
							  46,45,44,43,32,41,40};
	uint32_t i, j, cmp, start;
	start = 0; // don't shift an empty buffer;
	for (i=0;i<32;i++){
		cmp = 0x80;
		do{
			if (context[msb_order[i]] && cmp){
				//add
				struct_BN_shl(&buffer, &buffer);
				struct_BN_add(&buffer, &buffer, var_a);
				start = 1;
			}else{
				//shift
				if (start) // don't shift an empty buffer;
					struct_BN_shl(&buffer, &buffer);
			}
			cmp = cmp >> 1;
		}while(cmp>0);
	}
	res[0]=content[0];
	res[1]=content[1];
	res[2]=content[2];
}


void struct_BN_EC_Point_Add_Affine(const EC_GROUP *group, EC_POINT *r, const EC_POINT *a, const EC_POINT *b, BN_CTX *ctx)
{
/*
 * Jacobian addition of two affine points (X1, Y1, 1) and (X2, Y2, 1)
 *  nr Mul = 8

	if (X1 == X2)
   	   if (Y1 != Y2)
     	 return POINT_AT_INFINITY
   	   else
     	 return POINT_DOUBLE(X1, Y1, 1)
 	 H = X2 - X1
 	 H2 = H * H
 	 H3 = H2 * H
 	 R = Y2 - Y1
 	 R2 = R * R
 	 X3 = R2 - H3 - 2*X1*H2
 	 Y3 = R*(X1*H2 - X3) - Y1*H3
 	 Z3 = H

	return (X3, Y3, Z3)
 */


	_sidm_bn_context_t point_a[3];
	_sidm_bn_context_t point_b[3];
	_sidm_bn_context_t point_r[3];

	_sidm_bn_context_t loc_h2;
	_sidm_bn_context_t loc_h3;

	_sidm_bn_context_t loc_r;
	_sidm_bn_context_t loc_r2;

	_sidm_bn_context_t loc_calc;

	BIGNUM* X;
	BIGNUM* Y;
	BIGNUM* Z;

	BN_CTX_start(ctx);
	X = BN_CTX_get(ctx);
	Y = BN_CTX_get(ctx);
	Z = BN_CTX_get(ctx);
	EC_POINT_get_Jprojective_coordinates_GFp(group,a, X, Y, Z, ctx);
	BN_to_Struct(X,&point_a[0]);
	BN_to_Struct(Y,&point_a[1]);
	BN_to_Struct(Z,&point_a[2]);
	EC_POINT_get_Jprojective_coordinates_GFp(group,b, X, Y, Z, ctx);
	BN_to_Struct(X,&point_b[0]);
	BN_to_Struct(Y,&point_b[1]);
	BN_to_Struct(Z,&point_b[2]);

	// H = X2 - X1
	// Z3 = H
	struct_BN_sub(&point_r[2],&point_b[0],&point_a[0]);
	// H2 = H * H
	struct_BN_mul(&loc_h2,&point_r[2],&point_r[2]);
	// H3 = H2 * H
	struct_BN_mul(&loc_h3,&loc_h2,&point_r[2]);
	// R = Y2 - Y1
	struct_BN_sub(&loc_r,&point_b[1],&point_a[1]);
	// R2 = R * R
	struct_BN_mul(&loc_r2,&loc_r,&loc_r);

	// X3 = R2 - H3 - 2*X1*H2
	struct_BN_add(&loc_calc, &point_a[0],&point_a[0]);
	struct_BN_mul(&loc_calc,&loc_calc,&loc_h2);
	struct_BN_sub(&point_r[0],&loc_r2,&loc_h3);
	struct_BN_sub(&point_r[0],&point_r[0],&loc_calc);

	// 	 Y3 = R*(X1*H2 - X3) - Y1*H3
	struct_BN_mul(&loc_calc,&point_a[1],&loc_h3);
	struct_BN_mul(&point_r[1],&point_a[0],&loc_h2);
	struct_BN_sub(&point_r[1],&point_r[1],&point_r[0]);
	struct_BN_mul(&point_r[1],&point_r[1],&loc_r);
	struct_BN_sub(&point_r[1],&point_r[1],&loc_calc);

	Struct_to_BN(&point_r[0], X);
	Struct_to_BN(&point_r[1], Y);
	Struct_to_BN(&point_r[2], Z);

	EC_POINT_set_Jprojective_coordinates_GFp(group,r, X, Y, Z, ctx);
	BN_CTX_end(ctx);

}

size_t struct_EC_POINT_point2oct(const EC_GROUP *group, const EC_POINT *p,	point_conversion_form_t form,
        unsigned char *buf, size_t len, BN_CTX *ctx)
{
	BIGNUM* X;
	BIGNUM* Y;
	int length;

	BN_CTX_start(ctx);
	X = BN_CTX_get(ctx);
	Y = BN_CTX_get(ctx);

	EC_POINT_get_Jprojective_coordinates_GFp(group,p, X, Y, NULL, ctx);
	BN_CTX_end(ctx);
	length = BN_num_bytes(X);
	memset(buf,0,len);
	length = BN_bn2bin(X, buf+(33-length)); //(32-length_x)+1
	if (form == POINT_CONVERSION_UNCOMPRESSED){
		length = BN_num_bytes(Y);
		length = BN_bn2bin(Y, buf+(65-length)); // (64-length_y)+
		buf[0] = 0x04;
		length = 65;
	}else{
		if(BN_is_odd(Y)){
			buf[0] = 0x03;
		}else{
			buf[0] = 0x02;
		}
		length = 33;
	}
	return (length);
}

void BN_EC_Point_Add_Affine(const EC_GROUP *group, EC_POINT *r, const EC_POINT *a, const EC_POINT *b, BN_CTX *ctx)
{
/*
 * Jacobian addition of two affine points (X1, Y1, 1) and (X2, Y2, 1)
 */
	BIGNUM *X13,*Y13;

	BIGNUM *Y2;

	BIGNUM *Z3;

	BIGNUM *H23;
	BIGNUM *R1;

	BIGNUM *Calc;
	BIGNUM *field;

	BN_CTX_start(ctx);
	X13 = BN_CTX_get(ctx); Y13 = BN_CTX_get(ctx);
	Y2 = BN_CTX_get(ctx);
    Z3 = BN_CTX_get(ctx);
	H23 = BN_CTX_get(ctx); R1 = BN_CTX_get(ctx);
	Calc = BN_CTX_get(ctx); field = BN_CTX_get(ctx);
	EC_POINT_get_Jprojective_coordinates_GFp(group,a, X13, Y13, NULL, ctx);
	EC_POINT_get_Jprojective_coordinates_GFp(group,b, Z3, Y2, NULL, ctx);
	EC_GROUP_get_curve_GFp(group,field,NULL,NULL,ctx);
	// H = X2 - X1
	// Z3 = H
	BN_sub(Z3,Z3,X13);
	if (BN_is_negative(Z3))
		BN_add(Z3,Z3,field);
	// H2 = H * H
	// H3 = H2 * H
	BN_mod_sqr(H23,Z3,field,ctx);
	// R = Y2 - Y1
	BN_sub(R1,Y2,Y13);
	if (BN_is_negative(R1))
		BN_add(R1,R1,field);

	//Y3 X1*H2 (intermediate reusable result)
	BN_mod_mul(Y2,X13,H23,field,ctx);

	BN_mod_mul(H23,H23,Z3,field,ctx);
	// R2 = R * R --> intermediate result X3
	BN_mod_sqr(X13,R1,field,ctx);

	// X3 = R2 - H3 - 2*X1*H2
	BN_add(Calc,Y2,Y2);
	BN_add(Calc,Calc,H23);
	//	BN_mod_mul(Calc,Calc,H2,field,ctx);
	//	BN_mod_sub(X3,R2,H3,field,ctx);
	//	BN_mod_sub(X3,X3,Calc,field,ctx);
	BN_sub(X13,X13,Calc);
	if (BN_is_negative(X13))
		BN_add(X13,X13,field);

	// 	 Y3 = R*(X1*H2 - X3) - Y1*H3
	BN_mod_mul(Calc,Y13,H23,field,ctx);
	//	BN_mod_mul(Y3,X1,H2,field,ctx);
	BN_sub(Y13,Y2,X13);
	if (BN_is_negative(X13))
		BN_add(X13,X13,field);
	BN_mod_mul(Y13,Y13,R1,field,ctx);
	BN_sub(Y13,Y13,Calc);
	if (BN_is_negative(Y13))
		BN_add(Y13,Y13,field);

	EC_POINT_set_Jprojective_coordinates_GFp(group,r, X13, Y13, Z3, ctx);
	BN_CTX_end(ctx);

}

