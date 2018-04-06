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

#include "pattern.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <pthread.h>

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#include <pcre.h>

#include "util.h"
#include "avl.h"

/*
 * Common code for execution helper
 */

EC_KEY *
vg_exec_context_new_key(void)
{
	return EC_KEY_new_by_curve_name(NID_secp256k1);
}

/*
 * Thread synchronization helpers
 */

static pthread_mutex_t vg_thread_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t vg_thread_rdcond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t vg_thread_wrcond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t vg_thread_upcond = PTHREAD_COND_INITIALIZER;

static void
__vg_exec_context_yield(vg_exec_context_t *vxcp)
{
	vxcp->vxc_lockmode = 0;
	while (vxcp->vxc_vc->vc_thread_excl) {
		if (vxcp->vxc_stop) {
			assert(vxcp->vxc_vc->vc_thread_excl);
			vxcp->vxc_stop = 0;
			pthread_cond_signal(&vg_thread_upcond);
		}
		pthread_cond_wait(&vg_thread_rdcond, &vg_thread_lock);
	}
	assert(!vxcp->vxc_stop);
	assert(!vxcp->vxc_lockmode);
	vxcp->vxc_lockmode = 1;
}

int
vg_exec_context_upgrade_lock(vg_exec_context_t *vxcp)
{
	vg_exec_context_t *tp;
	vg_context_t *vcp;

	if (vxcp->vxc_lockmode == 2)
		return 0;

	pthread_mutex_lock(&vg_thread_lock);

	assert(vxcp->vxc_lockmode == 1);
	vxcp->vxc_lockmode = 0;
	vcp = vxcp->vxc_vc;

	if (vcp->vc_thread_excl++) {
		assert(vxcp->vxc_stop);
		vxcp->vxc_stop = 0;
		pthread_cond_signal(&vg_thread_upcond);
		pthread_cond_wait(&vg_thread_wrcond, &vg_thread_lock);

		for (tp = vcp->vc_threads; tp != NULL; tp = tp->vxc_next) {
			assert(!tp->vxc_lockmode);
			assert(!tp->vxc_stop);
		}

	} else {
		for (tp = vcp->vc_threads; tp != NULL; tp = tp->vxc_next) {
			if (tp->vxc_lockmode) {
				assert(tp->vxc_lockmode != 2);
				tp->vxc_stop = 1;
			}
		}

		do {
			for (tp = vcp->vc_threads;
			     tp != NULL;
			     tp = tp->vxc_next) {
				if (tp->vxc_lockmode) {
					assert(tp->vxc_lockmode != 2);
					pthread_cond_wait(&vg_thread_upcond,
							  &vg_thread_lock);
					break;
				}
			}
		} while (tp);
	}

	vxcp->vxc_lockmode = 2;
	pthread_mutex_unlock(&vg_thread_lock);
	return 1;
}

void
vg_exec_context_downgrade_lock(vg_exec_context_t *vxcp)
{
	pthread_mutex_lock(&vg_thread_lock);
	assert(vxcp->vxc_lockmode == 2);
	assert(!vxcp->vxc_stop);
	if (!--vxcp->vxc_vc->vc_thread_excl) {
		vxcp->vxc_lockmode = 1;
		pthread_cond_broadcast(&vg_thread_rdcond);
		pthread_mutex_unlock(&vg_thread_lock);
		return;
	}
	pthread_cond_signal(&vg_thread_wrcond);
	__vg_exec_context_yield(vxcp);
	pthread_mutex_unlock(&vg_thread_lock);
}

int
vg_exec_context_init(vg_context_t *vcp, vg_exec_context_t *vxcp)
{
	pthread_mutex_lock(&vg_thread_lock);

	memset(vxcp, 0, sizeof(*vxcp));

	vxcp->vxc_vc = vcp;

	BN_init(&vxcp->vxc_bntarg);
	BN_init(&vxcp->vxc_bnbase);
	BN_init(&vxcp->vxc_bntmp);
	BN_init(&vxcp->vxc_bntmp2);

	BN_set_word(&vxcp->vxc_bnbase, 58);

	vxcp->vxc_bnctx = BN_CTX_new();
	assert(vxcp->vxc_bnctx);
	vxcp->vxc_key = vg_exec_context_new_key();
	assert(vxcp->vxc_key);
	EC_KEY_precompute_mult(vxcp->vxc_key, vxcp->vxc_bnctx);

	vxcp->vxc_lockmode = 0;
	vxcp->vxc_stop = 0;

	vxcp->vxc_next = vcp->vc_threads;
	vcp->vc_threads = vxcp;
	vxcp->vc_combined_compressed = vcp->vc_compressed;
	__vg_exec_context_yield(vxcp);
	pthread_mutex_unlock(&vg_thread_lock);
	return 1;
}

void
vg_exec_context_del(vg_exec_context_t *vxcp)
{
	printf("vg_exec_context_del\n");
	
	vg_exec_context_t *tp, **pprev;

	if (vxcp->vxc_lockmode == 2)
		vg_exec_context_downgrade_lock(vxcp);

	pthread_mutex_lock(&vg_thread_lock);
	assert(vxcp->vxc_lockmode == 1);
	vxcp->vxc_lockmode = 0;

	for (pprev = &vxcp->vxc_vc->vc_threads, tp = *pprev;
	     (tp != vxcp) && (tp != NULL);
	     pprev = &tp->vxc_next, tp = *pprev);

	assert(tp == vxcp);
	*pprev = tp->vxc_next;

	if (tp->vxc_stop)
		pthread_cond_signal(&vg_thread_upcond);

	BN_clear_free(&vxcp->vxc_bntarg);
	BN_clear_free(&vxcp->vxc_bnbase);
	BN_clear_free(&vxcp->vxc_bntmp);
	BN_clear_free(&vxcp->vxc_bntmp2);
	BN_CTX_free(vxcp->vxc_bnctx);
	vxcp->vxc_bnctx = NULL;
	pthread_mutex_unlock(&vg_thread_lock);
}

void
vg_exec_context_yield(vg_exec_context_t *vxcp)
{
	if (vxcp->vxc_lockmode == 2)
		vg_exec_context_downgrade_lock(vxcp);

	else if (vxcp->vxc_stop) {
		assert(vxcp->vxc_lockmode == 1);
		pthread_mutex_lock(&vg_thread_lock);
		__vg_exec_context_yield(vxcp);
		pthread_mutex_unlock(&vg_thread_lock);
	}

	assert(vxcp->vxc_lockmode == 1);
}

void
vg_exec_context_consolidate_key(vg_exec_context_t *vxcp)
{
	if (vxcp->vxc_delta) {
                // privkey += delta
		BN_clear(&vxcp->vxc_bntmp);
		BN_set_word(&vxcp->vxc_bntmp, vxcp->vxc_delta);
		BN_add(&vxcp->vxc_bntmp2,
		       EC_KEY_get0_private_key(vxcp->vxc_key),
		       &vxcp->vxc_bntmp);
		vg_set_privkey(&vxcp->vxc_bntmp2, vxcp->vxc_key);
                // delta = 0
		vxcp->vxc_delta = 0;
	}
}

void
vg_exec_context_calc_address(vg_exec_context_t *vxcp)
{
	EC_POINT *pubkey;
	const EC_GROUP *pgroup;
	unsigned char eckey_buf[96], hash1[32], hash2[20];
	int len;

	vg_exec_context_consolidate_key(vxcp);
	pgroup = EC_KEY_get0_group(vxcp->vxc_key);
	pubkey = EC_POINT_new(pgroup);
	EC_POINT_copy(pubkey, EC_KEY_get0_public_key(vxcp->vxc_key));
	if (vxcp->vxc_vc->vc_pubkey_base) {
		EC_POINT_add(pgroup,
			     pubkey,
			     pubkey,
			     vxcp->vxc_vc->vc_pubkey_base,
			     vxcp->vxc_bnctx);
	}
	len = EC_POINT_point2oct(pgroup,
				 pubkey,
				 vxcp->vc_combined_compressed ? POINT_CONVERSION_COMPRESSED : POINT_CONVERSION_UNCOMPRESSED,
				 eckey_buf,
				 sizeof(eckey_buf),
				 vxcp->vxc_bnctx);
	SHA256(eckey_buf, len, hash1);
	RIPEMD160(hash1, sizeof(hash1), hash2);
	memcpy(&vxcp->vxc_binres[1],
	       hash2, 20);
	EC_POINT_free(pubkey);
}

enum {
	timing_hist_size = 5
};

typedef struct _timing_info_s {
	struct _timing_info_s	*ti_next;
	pthread_t		ti_thread;
	unsigned long		ti_last_rate;

	unsigned long long	ti_hist_time[timing_hist_size];
	unsigned long		ti_hist_work[timing_hist_size];
	int			ti_hist_last;
} timing_info_t;

static pthread_mutex_t timing_mutex = PTHREAD_MUTEX_INITIALIZER;

int
vg_output_timing(vg_context_t *vcp, int cycle, struct timeval *last)
{
	pthread_t me;
	struct timeval tvnow, tv;
	timing_info_t *tip, *mytip;
	unsigned long long rate, myrate = 0, mytime, total, sincelast;
	int p, i;

	/* Compute the rate */
	gettimeofday(&tvnow, NULL);
	timersub(&tvnow, last, &tv);
	memcpy(last, &tvnow, sizeof(*last));
	mytime = tv.tv_usec + (1000000ULL * tv.tv_sec);
	if (!mytime)
		mytime = 1;
	rate = 0;

	pthread_mutex_lock(&timing_mutex);
	me = pthread_self();
	for (tip = vcp->vc_timing_head, mytip = NULL;
	     tip != NULL; tip = tip->ti_next) {
		if (pthread_equal(tip->ti_thread, me)) {
			mytip = tip;
			p = ((tip->ti_hist_last + 1) % timing_hist_size);
			tip->ti_hist_time[p] = mytime;
			tip->ti_hist_work[p] = cycle;
			tip->ti_hist_last = p;

			mytime = 0;
			myrate = 0;
			for (i = 0; i < timing_hist_size; i++) {
				mytime += tip->ti_hist_time[i];
				myrate += tip->ti_hist_work[i];
			}
			myrate = (myrate * 1000000) / mytime;
			tip->ti_last_rate = myrate;
			rate += myrate;

		} else
			rate += tip->ti_last_rate;
	}
	if (!mytip) {
		mytip = (timing_info_t *) malloc(sizeof(*tip));
		mytip->ti_next = vcp->vc_timing_head;
		mytip->ti_thread = me;
		vcp->vc_timing_head = mytip;
		mytip->ti_hist_last = 0;
		mytip->ti_hist_time[0] = mytime;
		mytip->ti_hist_work[0] = cycle;
		for (i = 1; i < timing_hist_size; i++) {
			mytip->ti_hist_time[i] = 1;
			mytip->ti_hist_work[i] = 0;
		}
		myrate = ((unsigned long long)cycle * 1000000) / mytime;
		mytip->ti_last_rate = myrate;
		rate += myrate;
	}

	vcp->vc_timing_total += cycle;
	if (vcp->vc_timing_prevfound != vcp->vc_found) {
		vcp->vc_timing_prevfound = vcp->vc_found;
		vcp->vc_timing_sincelast = 0;
	}
	vcp->vc_timing_sincelast += cycle;

	if (mytip != vcp->vc_timing_head) {
		pthread_mutex_unlock(&timing_mutex);
		return myrate;
	}
	total = vcp->vc_timing_total;
	sincelast = vcp->vc_timing_sincelast;
	pthread_mutex_unlock(&timing_mutex);

	vcp->vc_output_timing(vcp, sincelast, rate, total);
	return myrate;
}

void
vg_context_thread_exit(vg_context_t *vcp)
{
	timing_info_t *tip, **ptip;
	pthread_t me;

	pthread_mutex_lock(&timing_mutex);
	me = pthread_self();
	for (ptip = &vcp->vc_timing_head, tip = *ptip;
	     tip != NULL;
	     ptip = &tip->ti_next, tip = *ptip) {
		if (!pthread_equal(tip->ti_thread, me))
			continue;
		*ptip = tip->ti_next;
		free(tip);
		break;
	}
	pthread_mutex_unlock(&timing_mutex);

}

static void
vg_timing_info_free(vg_context_t *vcp)
{
	timing_info_t *tp;
	while (vcp->vc_timing_head != NULL) {
		tp = vcp->vc_timing_head;
		vcp->vc_timing_head = tp->ti_next;
		free(tp);
	}
}

void
vg_output_timing_console(vg_context_t *vcp, double count,
			 unsigned long long rate, unsigned long long total)
{
	double targ;
	char *unit;
	char linebuf[80];
	int rem, p;

	targ = rate;
	unit = "key/s";
	if (targ > 1000) {
		unit = "Kkey/s";
		targ /= 1000.0;
		if (targ > 1000) {
			unit = "Mkey/s";
			targ /= 1000.0;
		}
	}

	rem = sizeof(linebuf);
	p = snprintf(linebuf, rem, "[%.2f %s][total %lld]",
		     targ, unit, total);
	assert(p > 0);
	rem -= p;
	if (rem < 0)
		rem = 0;

	if (rem) {
		memset(&linebuf[sizeof(linebuf)-rem], 0x20, rem);
		linebuf[sizeof(linebuf)-1] = '\0';
	}
	printf("\r%s", linebuf);
	fflush(stdout);
}

void
//vg_output_match_console(vg_context_t *vcp, EC_KEY *pkey, const char *pattern)
vg_output_match_console(vg_context_t *vcp, vg_exec_context_t *vxcp, const char *pattern)
{
	unsigned char key_buf[512], *pend;
	char addr_buf[64], addr2_buf[64];
	char privkey_buf[128];
	char * privkey_str = privkey_buf;
	const char *keytype = "Privkey";
	int len;
	int isscript = (vcp->vc_format == VCF_SCRIPT);
	EC_KEY *pkey = vxcp->vxc_key;
	EC_POINT *ppnt;
	int free_ppnt = 0;
	if (vcp->vc_pubkey_base) {
		ppnt = EC_POINT_new(EC_KEY_get0_group(pkey));
		EC_POINT_copy(ppnt, EC_KEY_get0_public_key(pkey));
		EC_POINT_add(EC_KEY_get0_group(pkey),
			     ppnt,
			     ppnt,
			     vcp->vc_pubkey_base,
			     NULL);
		free_ppnt = 1;
		keytype = "PrivkeyPart";
	} else {
		ppnt = (EC_POINT *) EC_KEY_get0_public_key(pkey);
	}

	assert(EC_KEY_check_key(pkey));
//	if (vcp->vc_combined_compressed)
	if (vxcp->vc_combined_compressed)
		vg_encode_address_compressed(ppnt,
				  EC_KEY_get0_group(pkey),
				  vcp->vc_pubkeytype, addr_buf);
	else
		vg_encode_address(ppnt,
				  EC_KEY_get0_group(pkey),
				  vcp->vc_pubkeytype, addr_buf);
	if (isscript)
		vg_encode_script_address(ppnt,
					 EC_KEY_get0_group(pkey),
					 vcp->vc_addrtype, addr2_buf);

	if (vcp->vc_key_protect_pass) {
		len = vg_protect_encode_privkey(privkey_buf,
						pkey, vcp->vc_privtype,
						VG_PROTKEY_DEFAULT,
						vcp->vc_key_protect_pass);
		if (len) {
			keytype = "Protkey";
		} else {
			fprintf(stderr,
				"ERROR: could not password-protect key\n");
			vcp->vc_key_protect_pass = NULL;
		}
	}
	if (!vcp->vc_key_protect_pass) {
//		if (vcp->vc_combined_compressed)
		privkey_str = BN_bn2hex(EC_KEY_get0_private_key(pkey));
		//printf("Privkey (hex): %s\n", privkey_str);
	/*
		if (vxcp->vc_combined_compressed)
			vg_encode_privkey_compressed(pkey, vcp->vc_privtype, privkey_buf);
		else
			vg_encode_privkey(pkey, vcp->vc_privtype, privkey_buf);
		*/
	}

	if (!vcp->vc_result_file || (vcp->vc_verbose > 1)) {
		//printf("\r%79s\rPattern: %s\n", "", pattern);
	}

	if (vcp->vc_verbose > 0) {
		if (vcp->vc_verbose > 1) {
			pend = key_buf;
			len = i2o_ECPublicKey(pkey, &pend);
			printf("Pubkey (hex): ");
			dumphex(key_buf, len);
			pend = key_buf;
			len = i2d_ECPrivateKey(pkey, &pend);
			printf("Privkey (ASN1): ");
			dumphex(key_buf, len);
		}

	}

	if (!vcp->vc_result_file || (vcp->vc_verbose > 0)) {
		if (isscript)
			printf("P2SHAddress: %s\n", addr2_buf);
		printf("\nAddress: %s\n"
		       "%s (hex): %s\n",
		       addr_buf, keytype, privkey_str);
	}

	if (vcp->vc_result_file) {
		FILE *fp = fopen(vcp->vc_result_file, "a");
		if (!fp) {
			fprintf(stderr,
				"ERROR: could not open result file: %s\n",
				strerror(errno));
		} else {
			if (isscript)
				fprintf(fp, "P2SHAddress: %s\n", addr2_buf);
			fprintf(fp,
				"Address: %s\n"
				"%s (hex): %s\n",
				addr_buf, keytype, privkey_str);
			fclose(fp);
		}
	}
	if (free_ppnt)
		EC_POINT_free(ppnt);
}


void
vg_context_free(vg_context_t *vcp)
{
	vg_timing_info_free(vcp);
	vcp->vc_free(vcp);
}

int
vg_context_add_patterns(vg_context_t *vcp,
			const char ** const patterns, int npatterns)
{
	printf("Searching for %d addresses\n", npatterns);
	vcp->patterns = (char*)calloc(20 /* address size */, npatterns);
        for (int i=0; i<npatterns; i++) {
                int len = strlen(patterns[i]);
                while (len) if (strchr(" \r\n\t", patterns[i][len-1])) len--; else break;
                if (len != 33 && len != 34) {
                        printf("Wrong address length at line %d: %d (must be 33 or 34)\n", i+1, len);
                        exit(-1);
                }
		char buf[21];
                if (!vg_b58_decode_check(patterns[i], buf, 21)) return 0;
		memcpy(vcp->patterns+i*20, buf+1, 20);
        }
        
	return vcp->vc_add_patterns(vcp, patterns, npatterns);
}

void
vg_context_clear_all_patterns(vg_context_t *vcp)
{
        free(vcp->patterns);
        vcp->patterns = NULL;
}

int
vg_context_start_threads(vg_context_t *vcp)
{
	vg_exec_context_t *vxcp;
	int res;

	for (vxcp = vcp->vc_threads; vxcp != NULL; vxcp = vxcp->vxc_next) {
        if (vcp->vc_verbose > 1) printf("\nStart a thread");
		res = pthread_create((pthread_t *) &vxcp->vxc_pthread,
				     NULL,
				     (void *(*)(void *)) vxcp->vxc_threadfunc,
				     vxcp);
		if (res) {
			fprintf(stderr, "ERROR: could not create thread: %d\n",
				res);
			vg_context_stop_threads(vcp);
			return -1;
		}
		vxcp->vxc_thread_active = 1;
	}
	return 0;
}

void
vg_context_stop_threads(vg_context_t *vcp)
{
	vcp->vc_halt = 1;
	vg_context_wait_for_completion(vcp);
	vcp->vc_halt = 0;
}

void
vg_context_wait_for_completion(vg_context_t *vcp)
{
	vg_exec_context_t *vxcp;

	for (vxcp = vcp->vc_threads; vxcp != NULL; vxcp = vxcp->vxc_next) {
		if (!vxcp->vxc_thread_active)
			continue;
		pthread_join((pthread_t) vxcp->vxc_pthread, NULL);
		vxcp->vxc_thread_active = 0;
	}
}


/*
 * Address prefix AVL tree node
 */

const int vpk_nwords = (25 + sizeof(BN_ULONG) - 1) / sizeof(BN_ULONG);

typedef struct _vg_prefix_s {
	avl_item_t		vp_item;
	struct _vg_prefix_s	*vp_sibling;
	const char		*vp_pattern;
} vg_prefix_t;

static void
vg_prefix_free(vg_prefix_t *vp)
{
	free(vp);
}

int str_cmp(const char * a, const char * b) {
	for (int i=0; a[i] || b[i]; i++) {
		if (a[i]<b[i]) return -1;
		if (a[i]>b[i]) return 1;
	}
	return 0;
}

static vg_prefix_t *
vg_prefix_avl_search(avl_root_t *rootp, char * pattern)
{
	vg_prefix_t *vp;
	avl_item_t *itemp = rootp->ar_root;

	while (itemp) {
		vp = avl_item_entry(itemp, vg_prefix_t, vp_item);
		int cmp = str_cmp(pattern, vp->vp_pattern);
		if (cmp < 0) {
			itemp = itemp->ai_left;
		} else {
			if (cmp > 0) {
				itemp = itemp->ai_right;
			} else
				return vp;
		}
	}
	return NULL;
}

static vg_prefix_t *
vg_prefix_avl_insert(avl_root_t *rootp, vg_prefix_t *vpnew)
{
	vg_prefix_t *vp;
	avl_item_t *itemp = NULL;
	avl_item_t **ptrp = &rootp->ar_root;

	while (*ptrp) {
		itemp = *ptrp;
//		ptrp = &itemp->ai_left;
		vp = avl_item_entry(itemp, vg_prefix_t, vp_item);
		int cmp = str_cmp(vpnew->vp_pattern, vp->vp_pattern);
		if (cmp < 0) {
			ptrp = &itemp->ai_left;
		} else {
			if (cmp > 0) {
				ptrp = &itemp->ai_right;
			} else
				return vp;
		}
	}

	vpnew->vp_item.ai_up = itemp;
	itemp = &vpnew->vp_item;
	*ptrp = itemp;
	avl_insert_fix(rootp, itemp);
	return NULL;
}

static vg_prefix_t *
vg_prefix_add(avl_root_t *rootp, const char *pattern)
{
	vg_prefix_t *vp, *vp2;
	vp = (vg_prefix_t *) malloc(sizeof(*vp));
	if (vp) {
		avl_item_init(&vp->vp_item);
		vp->vp_sibling = NULL;
		vp->vp_pattern = pattern;
		vp2 = vg_prefix_avl_insert(rootp, vp);
		if (vp2 != NULL) {
			fprintf(stderr,
				"Prefix '%s' ignored, overlaps '%s'\n",
				pattern, vp2->vp_pattern);
			vg_prefix_free(vp);
			vp = NULL;
		}
	}
	return vp;
}

static void
vg_prefix_delete(avl_root_t *rootp, vg_prefix_t *vp)
{
	vg_prefix_t *sibp, *delp;

	avl_remove(rootp, &vp->vp_item);
	sibp = vp->vp_sibling;
	while (sibp && sibp != vp) {
		avl_remove(rootp, &sibp->vp_item);
		delp = sibp;
		sibp = sibp->vp_sibling;
		vg_prefix_free(delp);
	}
	vg_prefix_free(vp);
}

typedef struct _vg_prefix_context_s {
	vg_context_t		base;
	avl_root_t		vcp_avlroot;
	BIGNUM			vcp_difficulty;
	int			vcp_caseinsensitive;
} vg_prefix_context_t;

void
vg_prefix_context_set_case_insensitive(vg_context_t *vcp, int caseinsensitive)
{
	((vg_prefix_context_t *) vcp)->vcp_caseinsensitive = caseinsensitive;
}

static void
vg_prefix_context_clear_all_patterns(vg_context_t *vcp)
{
	vg_prefix_context_t *vcpp = (vg_prefix_context_t *) vcp;
	vg_prefix_t *vp;
	unsigned long npfx_left = 0;

	while (!avl_root_empty(&vcpp->vcp_avlroot)) {
		vp = avl_item_entry(vcpp->vcp_avlroot.ar_root,
				    vg_prefix_t, vp_item);
		vg_prefix_delete(&vcpp->vcp_avlroot, vp);
		npfx_left++;
	}

	assert(npfx_left == vcpp->base.vc_npatterns);
	vcpp->base.vc_npatterns = 0;
	vcpp->base.vc_npatterns_start = 0;
	vcpp->base.vc_found = 0;
	BN_clear(&vcpp->vcp_difficulty);
}

static void
vg_prefix_context_free(vg_context_t *vcp)
{
	vg_prefix_context_t *vcpp = (vg_prefix_context_t *) vcp;
	vg_prefix_context_clear_all_patterns(vcp);
	BN_clear_free(&vcpp->vcp_difficulty);
	free(vcpp);
}

static int
vg_prefix_context_add_patterns(vg_context_t *vcp,
			       const char ** const patterns, int npatterns)
{
	vg_prefix_context_t *vcpp = (vg_prefix_context_t *) vcp;
	vg_prefix_t *vp;
	BN_CTX *bnctx;
	BIGNUM bntmp, bntmp2, bntmp3;
	int i;

	bnctx = BN_CTX_new();
	BN_init(&bntmp);
	BN_init(&bntmp2);
	BN_init(&bntmp3);

	for (i = 0; i < npatterns; i++) {
		vp = vg_prefix_add(&vcpp->vcp_avlroot, patterns[i]);
		if (!vp)
			continue;

	}

	vcpp->base.vc_npatterns = npatterns;

	BN_clear_free(&bntmp);
	BN_clear_free(&bntmp2);
	BN_clear_free(&bntmp3);
	BN_CTX_free(bnctx);
	return 1;
}

static int
vg_prefix_test(vg_exec_context_t *vxcp)
{
	vg_prefix_context_t *vcpp = (vg_prefix_context_t *) vxcp->vxc_vc;
	vg_prefix_t *vp;
	int res = 0;

	/*
	 * We constrain the prefix so that we can check for
	 * a match without generating the lower four byte
	 * check code.
	 */

	BN_bin2bn(vxcp->vxc_binres, 25, &vxcp->vxc_bntarg);
	char addr[35];
	vg_b58_encode_check(vxcp->vxc_binres, 21, addr);

research:
	vp = vg_prefix_avl_search(&vcpp->vcp_avlroot, addr);
	if (vp) {
		if (vg_exec_context_upgrade_lock(vxcp))
			goto research;

		vg_exec_context_consolidate_key(vxcp);
//		vcpp->base.vc_output_match(&vcpp->base, vxcp->vxc_key,
//					   vp->vp_pattern, &combined_compressed);
		vcpp->base.vc_output_match(&vcpp->base, vxcp,
					   vp->vp_pattern);

		vcpp->base.vc_found++;

		if (vcpp->base.vc_only_one) {
			return 2;
		}

		res = 1;
	}
	if (avl_root_empty(&vcpp->vcp_avlroot)) {
		return 2;
	}
	return res;
}

vg_context_t *
vg_prefix_context_new(int addrtype, int privtype, int caseinsensitive)
{
	vg_prefix_context_t *vcpp;

	vcpp = (vg_prefix_context_t *) malloc(sizeof(*vcpp));
	if (vcpp) {
		memset(vcpp, 0, sizeof(*vcpp));
		vcpp->base.vc_addrtype = addrtype;
		vcpp->base.vc_privtype = privtype;
		vcpp->base.vc_npatterns = 0;
		vcpp->base.vc_npatterns_start = 0;
		vcpp->base.vc_found = 0;
		vcpp->base.vc_chance = 0.0;
		vcpp->base.vc_free = vg_prefix_context_free;
		vcpp->base.vc_add_patterns = vg_prefix_context_add_patterns;
		vcpp->base.vc_clear_all_patterns =
			vg_prefix_context_clear_all_patterns;
		vcpp->base.vc_test = vg_prefix_test;
		vcpp->base.vc_hash160_sort = NULL;
		avl_root_init(&vcpp->vcp_avlroot);
		BN_init(&vcpp->vcp_difficulty);
		vcpp->vcp_caseinsensitive = caseinsensitive;
	}
	return &vcpp->base;
}




typedef struct _vg_regex_context_s {
	vg_context_t		base;
	pcre 			**vcr_regex;
	pcre_extra		**vcr_regex_extra;
	const char		**vcr_regex_pat;
	unsigned long		vcr_nalloc;
} vg_regex_context_t;

static int
vg_regex_context_add_patterns(vg_context_t *vcp,
			      const char ** const patterns, int npatterns)
{
	vg_regex_context_t *vcrp = (vg_regex_context_t *) vcp;
	const char *pcre_errptr;
	int pcre_erroffset;
	unsigned long i, nres, count;
	void **mem;

	if (!npatterns)
		return 1;

	if (npatterns > (vcrp->vcr_nalloc - vcrp->base.vc_npatterns)) {
		count = npatterns + vcrp->base.vc_npatterns;
		if (count < (2 * vcrp->vcr_nalloc)) {
			count = (2 * vcrp->vcr_nalloc);
		}
		if (count < 16) {
			count = 16;
		}
		mem = (void **) malloc(3 * count * sizeof(void*));
		if (!mem)
			return 0;

		for (i = 0; i < vcrp->base.vc_npatterns; i++) {
			mem[i] = vcrp->vcr_regex[i];
			mem[count + i] = vcrp->vcr_regex_extra[i];
			mem[(2 * count) + i] = (void *) vcrp->vcr_regex_pat[i];
		}

		if (vcrp->vcr_nalloc)
			free(vcrp->vcr_regex);
		vcrp->vcr_regex = (pcre **) mem;
		vcrp->vcr_regex_extra = (pcre_extra **) &mem[count];
		vcrp->vcr_regex_pat = (const char **) &mem[2 * count];
		vcrp->vcr_nalloc = count;
	}

	nres = vcrp->base.vc_npatterns;
	for (i = 0; i < npatterns; i++) {
		vcrp->vcr_regex[nres] =
			pcre_compile(patterns[i], 0,
				     &pcre_errptr, &pcre_erroffset, NULL);
		if (!vcrp->vcr_regex[nres]) {
			const char *spaces = "                ";
			fprintf(stderr, "%s\n", patterns[i]);
			while (pcre_erroffset > 16) {
				fprintf(stderr, "%s", spaces);
				pcre_erroffset -= 16;
			}
			if (pcre_erroffset > 0)
				fprintf(stderr,
					"%s", &spaces[16 - pcre_erroffset]);
			fprintf(stderr, "^\nRegex error: %s\n", pcre_errptr);
			continue;
		}
		vcrp->vcr_regex_extra[nres] =
			pcre_study(vcrp->vcr_regex[nres], 0, &pcre_errptr);
		if (pcre_errptr) {
			fprintf(stderr, "Regex error: %s\n", pcre_errptr);
			pcre_free(vcrp->vcr_regex[nres]);
			continue;
		}
		vcrp->vcr_regex_pat[nres] = patterns[i];
		nres += 1;
	}

	if (nres == vcrp->base.vc_npatterns)
		return 0;

	vcrp->base.vc_npatterns_start += (nres - vcrp->base.vc_npatterns);
	vcrp->base.vc_npatterns = nres;
	return 1;
}

static void
vg_regex_context_clear_all_patterns(vg_context_t *vcp)
{
	vg_regex_context_t *vcrp = (vg_regex_context_t *) vcp;
	int i;
	for (i = 0; i < vcrp->base.vc_npatterns; i++) {
		if (vcrp->vcr_regex_extra[i])
			pcre_free(vcrp->vcr_regex_extra[i]);
		pcre_free(vcrp->vcr_regex[i]);
	}
	vcrp->base.vc_npatterns = 0;
	vcrp->base.vc_npatterns_start = 0;
	vcrp->base.vc_found = 0;
}

static void
vg_regex_context_free(vg_context_t *vcp)
{
	vg_regex_context_t *vcrp = (vg_regex_context_t *) vcp;
	vg_regex_context_clear_all_patterns(vcp);
	if (vcrp->vcr_nalloc)
		free(vcrp->vcr_regex);
	free(vcrp);
}

static int
vg_regex_test(vg_exec_context_t *vxcp)
{
	vg_regex_context_t *vcrp = (vg_regex_context_t *) vxcp->vxc_vc;

	unsigned char hash1[32], hash2[32];
	int i, zpfx, p, d, nres, re_vec[9];
	char b58[40];
	BIGNUM bnrem;
	BIGNUM *bn, *bndiv, *bnptmp;
	int res = 0;

	pcre *re;

	BN_init(&bnrem);

	/* Hash the hash and write the four byte check code */
	SHA256(vxcp->vxc_binres, 21, hash1);
	SHA256(hash1, sizeof(hash1), hash2);
	memcpy(&vxcp->vxc_binres[21], hash2, 4);

	bn = &vxcp->vxc_bntmp;
	bndiv = &vxcp->vxc_bntmp2;

	BN_bin2bn(vxcp->vxc_binres, 25, bn);

	/* Compute the complete encoded address */
	for (zpfx = 0; zpfx < 25 && vxcp->vxc_binres[zpfx] == 0; zpfx++);
	p = sizeof(b58) - 1;
	b58[p] = '\0';
	while (!BN_is_zero(bn)) {
		BN_div(bndiv, &bnrem, bn, &vxcp->vxc_bnbase, vxcp->vxc_bnctx);
		bnptmp = bn;
		bn = bndiv;
		bndiv = bnptmp;
		d = BN_get_word(&bnrem);
		b58[--p] = vg_b58_alphabet[d];
	}
	while (zpfx--) {
		b58[--p] = vg_b58_alphabet[0];
	}

	/*
	 * Run the regular expressions on it
	 * SLOW, runs in linear time with the number of REs
	 */
restart_loop:
	nres = vcrp->base.vc_npatterns;
	if (!nres) {
		res = 2;
		goto out;
	}
	for (i = 0; i < nres; i++) {
		d = pcre_exec(vcrp->vcr_regex[i],
			      vcrp->vcr_regex_extra[i],
			      &b58[p], (sizeof(b58) - 1) - p, 0,
			      0,
			      re_vec, sizeof(re_vec)/sizeof(re_vec[0]));

		if (d <= 0) {
			if (d != PCRE_ERROR_NOMATCH) {
				fprintf(stderr, "PCRE error: %d\n", d);
				res = 2;
				goto out;
			}
			continue;
		}

		re = vcrp->vcr_regex[i];

		if (vg_exec_context_upgrade_lock(vxcp) &&
		    ((i >= vcrp->base.vc_npatterns) ||
		     (vcrp->vcr_regex[i] != re)))
			goto restart_loop;

		vg_exec_context_consolidate_key(vxcp);
//		vcrp->base.vc_output_match(&vcrp->base, vxcp->vxc_key,
//					   vcrp->vcr_regex_pat[i], combined_compressed);
		vcrp->base.vc_output_match(&vcrp->base, vxcp,
					   vcrp->vcr_regex_pat[i]);
		vcrp->base.vc_found++;

		if (vcrp->base.vc_only_one) {
			res = 2;
			goto out;
		}

		if (vcrp->base.vc_remove_on_match) {
			pcre_free(vcrp->vcr_regex[i]);
			if (vcrp->vcr_regex_extra[i])
				pcre_free(vcrp->vcr_regex_extra[i]);
			nres -= 1;
			vcrp->base.vc_npatterns = nres;
			if (!nres) {
				res = 2;
				goto out;
			}
			vcrp->vcr_regex[i] = vcrp->vcr_regex[nres];
			vcrp->vcr_regex_extra[i] =
				vcrp->vcr_regex_extra[nres];
			vcrp->vcr_regex_pat[i] = vcrp->vcr_regex_pat[nres];
			vcrp->base.vc_npatterns = nres;
			vcrp->base.vc_pattern_generation++;
		}
		res = 1;
	}
out:
	BN_clear_free(&bnrem);
	return res;
}

vg_context_t *
vg_regex_context_new(int addrtype, int privtype)
{
	vg_regex_context_t *vcrp;

	vcrp = (vg_regex_context_t *) malloc(sizeof(*vcrp));
	if (vcrp) {
		memset(vcrp, 0, sizeof(*vcrp));
		vcrp->base.vc_addrtype = addrtype;
		vcrp->base.vc_privtype = privtype;
		vcrp->base.vc_npatterns = 0;
		vcrp->base.vc_npatterns_start = 0;
		vcrp->base.vc_found = 0;
		vcrp->base.vc_chance = 0.0;
		vcrp->base.vc_free = vg_regex_context_free;
		vcrp->base.vc_add_patterns = vg_regex_context_add_patterns;
		vcrp->base.vc_clear_all_patterns =
			vg_regex_context_clear_all_patterns;
		vcrp->base.vc_test = vg_regex_test;
		vcrp->base.vc_hash160_sort = NULL;
		vcrp->vcr_regex = NULL;
		vcrp->vcr_nalloc = 0;
	}
	return &vcrp->base;
}
