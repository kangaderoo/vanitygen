/*
 * Vanitygen, vanity bitcoin address generator
 * Copyright (C) 2011 <samr7@cs.washington.edu>
 *
 * Vanitygen is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version. 
 *
 * Vanitygen is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Vanitygen.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <pthread.h>

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/rand.h>

#include "pattern.h"
#include "util.h"
#include "rmd160.h"
#include "sha256.h"

#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

const char *version = VANITYGEN_VERSION;


/*
 * Address search thread main loop
 */

void *
vg_thread_loop(void *arg)
{
	unsigned char hash_buf[128*4]  				__attribute__((aligned(16)));;
	unsigned char hash_buf_transpose[128*4]  	__attribute__((aligned(16)));;
	// keep the size 128, in order to maintain the SHA256 512 bits per chunk
	uint32_t *sha256lenPtr = &hash_buf;
	unsigned char *eckey_buf;
	unsigned char hash1[32*4] 					__attribute__((aligned(16)));;;
	unsigned char hash2[32*4] 					__attribute__((aligned(16)));;;
	unsigned char hash1_transpose[32*4] 		__attribute__((aligned(16)));;;
	unsigned char hash2_transpose[32*4] 		__attribute__((aligned(16)));;;

	int i, j, c, len, output_interval;
	int hash_len;
	const step = 4;

	const BN_ULONG rekey_max = 10000000;
	BN_ULONG npoints, rekey_at, nbatch;

	vg_context_t *vcp = (vg_context_t *) arg;
	EC_KEY *pkey = NULL;
	const EC_GROUP *pgroup;
	const EC_POINT *pgen;
	const int ptarraysize = 256;
	EC_POINT *ppnt[ptarraysize];
	EC_POINT *pbatchinc;

	vg_test_func_t test_func = vcp->vc_test;
	vg_exec_context_t ctx;
	vg_exec_context_t *vxcp;

	struct timeval tvstart;

//	const __m128i vm = _mm_setr_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
//	const __m128i vm = _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);

	uint32_t         MDbuf[8 * 4]  __attribute__((aligned(16)));
	uint32_t         MDbuf_transpose[8 * 4]  __attribute__((aligned(16)));
//    __m128i 		 *MDbufPtr = (__m128i*) MDbuf;
    unsigned char    *MDBufChar = (unsigned char*) MDbuf;
//    BN_CTX           *ctx_buffer[4];

	//	dword         MDBufByte[5] __attribute__((aligned(16)));

	memset(&ctx,                0, sizeof(ctx));
	memset(&hash_buf,           0, 4*128);
	memset(&hash_buf_transpose, 0, 4*128);
	memset(&hash2_transpose,    0 ,4*32);

	vxcp = &ctx;

	vg_exec_context_init(vcp, &ctx);

	pkey = vxcp->vxc_key;
	pgroup = EC_KEY_get0_group(pkey);
	pgen = EC_GROUP_get0_generator(pgroup);

	for (i = 0; i < ptarraysize; i++) {
		ppnt[i] = EC_POINT_new(pgroup);
		if (!ppnt[i]) {
			fprintf(stderr, "ERROR: out of memory?\n");
			exit(1);
		}
	}
	pbatchinc = EC_POINT_new(pgroup);
	if (!pbatchinc) {
		fprintf(stderr, "ERROR: out of memory?\n");
		exit(1);
	}

	BN_set_word(&vxcp->vxc_bntmp, ptarraysize);
	EC_POINT_mul(pgroup, pbatchinc, &vxcp->vxc_bntmp, NULL, NULL,
		     vxcp->vxc_bnctx);
	EC_POINT_make_affine(pgroup, pbatchinc, vxcp->vxc_bnctx);

	npoints = 0;
	rekey_at = 0;
	nbatch = 0;
	vxcp->vxc_key = pkey;
	vxcp->vxc_binres[0] = vcp->vc_addrtype;
	c = 0;
	output_interval = 1000;
	gettimeofday(&tvstart, NULL);

	if (vcp->vc_format == VCF_SCRIPT) {
		for (j=0;j<4;j++){
			hash_buf[ 0+j*128] = 0x51;  // OP_1
			hash_buf[ 1+j*128] = 0x41;  // pubkey length
			// gap for pubkey
			hash_buf[67+j*128] = 0x51;  // OP_1
			hash_buf[68+j*128] = 0xae;  // OP_CHECKMULTISIG
		}
		eckey_buf = hash_buf + 2;
		hash_len = 69;

	} else {
		eckey_buf = hash_buf;
		hash_len = 65;
	}

	while (!vcp->vc_halt) {
		if (++npoints >= rekey_at) {
			vg_exec_context_upgrade_lock(vxcp);
			/* Generate a new random private key */
			EC_KEY_generate_key(pkey);
			npoints = 0;

			/* Determine rekey interval */
			EC_GROUP_get_order(pgroup, &vxcp->vxc_bntmp,
					   vxcp->vxc_bnctx);
			BN_sub(&vxcp->vxc_bntmp2,
			       &vxcp->vxc_bntmp,
			       EC_KEY_get0_private_key(pkey));
			rekey_at = BN_get_word(&vxcp->vxc_bntmp2);
			if ((rekey_at == BN_MASK2) || (rekey_at > rekey_max))
				rekey_at = rekey_max;
			assert(rekey_at > 0);

			EC_POINT_copy(ppnt[0], EC_KEY_get0_public_key(pkey));
			vg_exec_context_downgrade_lock(vxcp);

			npoints++;
			vxcp->vxc_delta = 0;

			if (vcp->vc_pubkey_base)
				EC_POINT_add(pgroup,
					     ppnt[0],
					     ppnt[0],
					     vcp->vc_pubkey_base,
					     vxcp->vxc_bnctx);

			for (nbatch = 1;
			     (nbatch < ptarraysize) && (npoints < rekey_at);
			     nbatch++, npoints++) {
				EC_POINT_add(pgroup,
					     ppnt[nbatch],
					     ppnt[nbatch-1],
					     pgen, vxcp->vxc_bnctx);
			}

		} else {
			/*
			 * Common case
			 *
			 * EC_POINT_add() can skip a few multiplies if
			 * one or both inputs are affine (Z_is_one).
			 * This is the case for every point in ppnt, as
			 * well as pbatchinc.
			 */
			assert(nbatch == ptarraysize);
			for (nbatch = 0;
			     (nbatch < ptarraysize) && (npoints < rekey_at);
			     nbatch++, npoints++) {
				EC_POINT_add(pgroup,
					     ppnt[nbatch],
					     ppnt[nbatch],
					     pbatchinc,
					     vxcp->vxc_bnctx);
			}
		}

		/*
		 * The single most expensive operation performed in this
		 * loop is modular inversion of ppnt->Z.  There is an
		 * algorithm implemented in OpenSSL to do batched inversion
		 * that only does one actual BN_mod_inverse(), and saves
		 * a _lot_ of time.
		 *
		 * To take advantage of this, we batch up a few points,
		 * and feed them to EC_POINTs_make_affine() below.
		 */

		EC_POINTs_make_affine(pgroup, nbatch, ppnt, vxcp->vxc_bnctx);

		for (i = 0; i < nbatch; i=i+step) {
			for (j=0; j< step;j++){
				/* Hash the public key */
				len = EC_POINT_point2oct(pgroup, ppnt[i+j],
							 POINT_CONVERSION_UNCOMPRESSED,
							 eckey_buf+(j*128),
							 65,
//							 ctx_buffer[j]);
							 vxcp->vxc_bnctx);
				assert(len == 65);
				vxcp->vxc_delta++;
			}

#if 0
			for (j=0; j< step;j++){
				SHA256(hash_buf+(j*128), hash_len, hash1+(j*32));
			}
#else
			for (j=0; j< step;j++){
				// hash_len is 65 or 69 length, so for SHA256 always two chunks
				// so the SHA prepare is here; add "1" and length are inserted in the buffer
				hash_buf[hash_len+(j*128)]= 0x80;
				sha256lenPtr[30+j*32] = (hash_len >> 29);
				sha256lenPtr[31+j*32] =	hash_len << 3;
			}
			// transpose the hash_buf from row to column
			MM_matrix_transpose_r2c(hash_buf,hash_buf_transpose, 4, 32);
			// Big/small endian recoding
			// don't use 32, the last two positions hold the length, already formatted correctly.
			// since the buffer also contains 0's minimal = 18 (69/4)+1, big endians of 0 are still 0
			MM_beRecode(hash_buf_transpose,30);
			// init the hash
			MM_sha256_init(hash1);
			// run transform first chunk
			MM_sha256_transform(hash1, hash_buf_transpose);
			// run transform 2nd chunk
			MM_sha256_transform(hash1, hash_buf_transpose+256);
			// Big/small endian recoding
			MM_beRecode(hash1,16);

#endif
			if (step==1){
				MDinit(MDbuf);
				MDfinish(MDbuf, hash1, sizeof(hash1)/4, 0);
			}else{
#if 0
				MDinit(MDbuf);
				MDfinish(MDbuf, hash1, sizeof(hash1)/4, 0);
				MDinit(MDbuf+8);
				MDfinish(MDbuf+8, hash1+32, sizeof(hash1)/4, 0);
				MDinit(MDbuf+16);
				MDfinish(MDbuf+16, hash1+64, sizeof(hash1)/4, 0);
				MDinit(MDbuf+24);
				MDfinish(MDbuf+24, hash1+96, sizeof(hash1)/4, 0);
#else
//				MD_matrix_transpose_r2c(hash1,hash1_transpose, 4, 8);
				MM_MDinit(MDbuf_transpose);
//				_mm_MDfinish(MDbuf_transpose, hash1_transpose /*hash1*/, sizeof(hash1)/4, 0);
				MM_MDfinish(MDbuf_transpose, hash1 /*hash1_transpose*/, sizeof(hash2)/4, 0);
				MM_matrix_transpose_c2r(MDbuf_transpose, MDbuf, 8, 4);
#endif
			}
			vxcp->vxc_delta=vxcp->vxc_delta-step;
		    for (j=0;j<step;j++){
	    		memcpy(vxcp->vxc_binres+1,MDBufChar+j*32,20);
		    	switch (test_func(vxcp)) {
					case 1:
						npoints = 0;
						rekey_at = 0;
						i = nbatch;
						j = step;
						break;
					case 2:
						goto out;
					default:
						break;
				}
			    vxcp->vxc_delta++;
		    }
		}
		c += i;
		if (c >= output_interval) {
			output_interval = vg_output_timing(vcp, c, &tvstart);
			if (output_interval > 250000)
				output_interval = 250000;
			c = 0;
		}

		vg_exec_context_yield(vxcp);
	}

out:

	vg_exec_context_del(&ctx);
	vg_context_thread_exit(vcp);


	for (i = 0; i < ptarraysize; i++)
		if (ppnt[i])
			EC_POINT_free(ppnt[i]);
	if (pbatchinc)
		EC_POINT_free(pbatchinc);
	return NULL;
}


#if !defined(_WIN32)
int
count_processors(void)
{
	FILE *fp;
	char buf[512];
	int count = 0;

	fp = fopen("/proc/cpuinfo", "r");
	if (!fp)
		return -1;

	while (fgets(buf, sizeof(buf), fp)) {
		if (!strncmp(buf, "processor\t", 10))
			count += 1;
	}
	fclose(fp);
	return count;
}
#endif

int
start_threads(vg_context_t *vcp, int nthreads)
{
	pthread_t thread;

	if (nthreads <= 0) {
		/* Determine the number of threads */
		nthreads = count_processors();
		if (nthreads <= 0) {
			fprintf(stderr,
				"ERROR: could not determine processor count\n");
			nthreads = 1;
		}
	}

	if (vcp->vc_verbose > 1) {
		fprintf(stderr, "Using %d worker thread(s)\n", nthreads);
	}

	while (--nthreads) {
		if (pthread_create(&thread, NULL, vg_thread_loop, vcp))
			return 0;
	}

	vg_thread_loop(vcp);
	return 1;
}


void
usage(const char *name)
{
	fprintf(stderr,
"Vanitygen %s (" OPENSSL_VERSION_TEXT ")\n"
"Usage: %s [-vqnrik1NT] [-t <threads>] [-f <filename>|-] [<pattern>...]\n"
"Generates a bitcoin receiving address matching <pattern>, and outputs the\n"
"address and associated private key.  The private key may be stored in a safe\n"
"location or imported into a bitcoin client to spend any balance received on\n"
"the address.\n"
"By default, <pattern> is interpreted as an exact prefix.\n"
"\n"
"Options:\n"
"-v            Verbose output\n"
"-q            Quiet output\n"
"-n            Simulate\n"
"-r            Use regular expression match instead of prefix\n"
"              (Feasibility of expression is not checked)\n"
"-i            Case-insensitive prefix search\n"
"-k            Keep pattern and continue search after finding a match\n"
"-1            Stop after first match\n"
"-N            Generate namecoin address\n"
"-T            Generate bitcoin testnet address\n"
"-X <version>  Generate address with the given version\n"
"-F <format>   Generate address with the given format (pubkey or script)\n"
"-P <pubkey>   Specify base public key for piecewise key generation\n"
"-e            Encrypt private keys, prompt for password\n"
"-E <password> Encrypt private keys with <password> (UNSAFE)\n"
"-t <threads>  Set number of worker threads (Default: number of CPUs)\n"
"-f <file>     File containing list of patterns, one per line\n"
"              (Use \"-\" as the file name for stdin)\n"
"-o <file>     Write pattern matches to <file>\n"
"-s <file>     Seed random number generator from <file>\n",
version, name);
}

#define MAX_FILE 4

int
main(int argc, char **argv)
{
	int addrtype = 0;
	int scriptaddrtype = 5;
	int privtype = 128;
	int pubkeytype;
	enum vg_format format = VCF_PUBKEY;
	int regex = 0;
	int caseinsensitive = 0;
	int verbose = 1;
	int simulate = 0;
	int remove_on_match = 1;
	int only_one = 0;
	int prompt_password = 0;
	int opt;
	char *seedfile = NULL;
	char pwbuf[128];
	const char *result_file = NULL;
	const char *key_password = NULL;
	char **patterns;
	int npatterns = 0;
	int nthreads = 0;
	vg_context_t *vcp = NULL;
	EC_POINT *pubkey_base = NULL;

	FILE *pattfp[MAX_FILE], *fp;
	int pattfpi[MAX_FILE];
	int npattfp = 0;
	int pattstdin = 0;

	int i;

	while ((opt = getopt(argc, argv, "vqnrik1eE:P:NTX:F:t:h?f:o:s:")) != -1) {
		switch (opt) {
		case 'v':
			verbose = 2;
			break;
		case 'q':
			verbose = 0;
			break;
		case 'n':
			simulate = 1;
			break;
		case 'r':
			regex = 1;
			break;
		case 'i':
			caseinsensitive = 1;
			break;
		case 'k':
			remove_on_match = 0;
			break;
		case '1':
			only_one = 1;
			break;
		case 'N':
			addrtype = 52;
			privtype = 180;
			scriptaddrtype = -1;
			break;
		case 'T':
			addrtype = 111;
			privtype = 239;
			scriptaddrtype = 196;
			break;
		case 'X':
			addrtype = atoi(optarg);
			privtype = 128 + addrtype;
			scriptaddrtype = addrtype;
			break;
		case 'F':
			if (!strcmp(optarg, "script"))
				format = VCF_SCRIPT;
			else
			if (strcmp(optarg, "pubkey")) {
				fprintf(stderr,
					"Invalid format '%s'\n", optarg);
				return 1;
			}
			break;
		case 'P': {
			if (pubkey_base != NULL) {
				fprintf(stderr,
					"Multiple base pubkeys specified\n");
				return 1;
			}
			EC_KEY *pkey = vg_exec_context_new_key();
			pubkey_base = EC_POINT_hex2point(
				EC_KEY_get0_group(pkey),
				optarg, NULL, NULL);
			EC_KEY_free(pkey);
			if (pubkey_base == NULL) {
				fprintf(stderr,
					"Invalid base pubkey\n");
				return 1;
			}
			break;
		}
			
		case 'e':
			prompt_password = 1;
			break;
		case 'E':
			key_password = optarg;
			break;
		case 't':
			nthreads = atoi(optarg);
			if (nthreads == 0) {
				fprintf(stderr,
					"Invalid thread count '%s'\n", optarg);
				return 1;
			}
			break;
		case 'f':
			if (npattfp >= MAX_FILE) {
				fprintf(stderr,
					"Too many input files specified\n");
				return 1;
			}
			if (!strcmp(optarg, "-")) {
				if (pattstdin) {
					fprintf(stderr, "ERROR: stdin "
						"specified multiple times\n");
					return 1;
				}
				fp = stdin;
			} else {
				fp = fopen(optarg, "r");
				if (!fp) {
					fprintf(stderr,
						"Could not open %s: %s\n",
						optarg, strerror(errno));
					return 1;
				}
			}
			pattfp[npattfp] = fp;
			pattfpi[npattfp] = caseinsensitive;
			npattfp++;
			break;
		case 'o':
			if (result_file) {
				fprintf(stderr,
					"Multiple output files specified\n");
				return 1;
			}
			result_file = optarg;
			break;
		case 's':
			if (seedfile != NULL) {
				fprintf(stderr,
					"Multiple RNG seeds specified\n");
				return 1;
			}
			seedfile = optarg;
			break;
		default:
			usage(argv[0]);
			return 1;
		}
	}

#if OPENSSL_VERSION_NUMBER < 0x10000000L
	/* Complain about older versions of OpenSSL */
	if (verbose > 0) {
		fprintf(stderr,
			"WARNING: Built with " OPENSSL_VERSION_TEXT "\n"
			"WARNING: Use OpenSSL 1.0.0d+ for best performance\n");
	}
#endif

	if (caseinsensitive && regex)
		fprintf(stderr,
			"WARNING: case insensitive mode incompatible with "
			"regular expressions\n");

	pubkeytype = addrtype;
	if (format == VCF_SCRIPT)
	{
		if (scriptaddrtype == -1)
		{
			fprintf(stderr,
				"Address type incompatible with script format\n");
			return 1;
		}
		addrtype = scriptaddrtype;
	}

	if (seedfile) {
		opt = -1;
#if !defined(_WIN32)
		{	struct stat st;
			if (!stat(seedfile, &st) &&
			    (st.st_mode & (S_IFBLK|S_IFCHR))) {
				opt = 32;
		} }
#endif
		opt = RAND_load_file(seedfile, opt);
		if (!opt) {
			fprintf(stderr, "Could not load RNG seed %s\n", optarg);
			return 1;
		}
		if (verbose > 0) {
			fprintf(stderr,
				"Read %d bytes from RNG seed file\n", opt);
		}
	}

	if (regex) {
		vcp = vg_regex_context_new(addrtype, privtype);

	} else {
		vcp = vg_prefix_context_new(addrtype, privtype,
					    caseinsensitive);
	}

	vcp->vc_verbose = verbose;
	vcp->vc_result_file = result_file;
	vcp->vc_remove_on_match = remove_on_match;
	vcp->vc_only_one = only_one;
	vcp->vc_format = format;
	vcp->vc_pubkeytype = pubkeytype;
	vcp->vc_pubkey_base = pubkey_base;

	vcp->vc_output_match = vg_output_match_console;
	vcp->vc_output_timing = vg_output_timing_console;

	if (!npattfp) {
		if (optind >= argc) {
			usage(argv[0]);
			return 1;
		}
		patterns = &argv[optind];
		npatterns = argc - optind;

		if (!vg_context_add_patterns(vcp,
					     (const char ** const) patterns,
					     npatterns))
		return 1;
	}

	for (i = 0; i < npattfp; i++) {
		fp = pattfp[i];
		if (!vg_read_file(fp, &patterns, &npatterns)) {
			fprintf(stderr, "Failed to load pattern file\n");
			return 1;
		}
		if (fp != stdin)
			fclose(fp);

		if (!regex)
			vg_prefix_context_set_case_insensitive(vcp, pattfpi[i]);

		if (!vg_context_add_patterns(vcp,
					     (const char ** const) patterns,
					     npatterns))
		return 1;
	}

	if (!vcp->vc_npatterns) {
		fprintf(stderr, "No patterns to search\n");
		return 1;
	}

	if (prompt_password) {
		if (!vg_read_password(pwbuf, sizeof(pwbuf)))
			return 1;
		key_password = pwbuf;
	}
	vcp->vc_key_protect_pass = key_password;
	if (key_password) {
		if (!vg_check_password_complexity(key_password, verbose))
			fprintf(stderr,
				"WARNING: Protecting private keys with "
				"weak password\n");
	}

	if ((verbose > 0) && regex && (vcp->vc_npatterns > 1))
		fprintf(stderr,
			"Regular expressions: %ld\n", vcp->vc_npatterns);

	if (simulate)
		return 0;

	if (!start_threads(vcp, nthreads))
		return 1;
	return 0;
}
