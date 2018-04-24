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

#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/hmac.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif
#else
#include <CL/cl.h>
#endif

#include "oclengine.h"
#include "pattern.h"
#include "util.h"
#include <assert.h>

#define MAX_SLOT 2
#define MAX_ARG 10
#define MAX_KERNEL 4

#define is_pow2(v) (!((v) & ((v)-1)))
#define round_up_pow2(x, a) (((x) + ((a)-1)) & ~((a)-1))

void vg_ocl_free_args(vg_ocl_context_t *vocp);
void *vg_opencl_loop(vg_exec_context_t *arg);
int vg_prefix_test(vg_exec_context_t *vxcp);
const EC_GROUP *pgroup;

/* OpenCL address searching mode */
struct _vg_ocl_context_s;
typedef int (*vg_ocl_init_t)(struct _vg_ocl_context_s *);
typedef int (*vg_ocl_check_t)(struct _vg_ocl_context_s *, int slot);

struct _vg_ocl_context_s {
	vg_exec_context_t		base;
	cl_device_id			voc_ocldid;
	cl_context			voc_oclctx;
	cl_command_queue		voc_oclcmdq;
	cl_program			voc_oclprog;
	vg_ocl_init_t			voc_init_func;
	vg_ocl_init_t			voc_rekey_func;
	int				voc_quirks;
	int				voc_nslots;
	cl_kernel			voc_oclkernel[MAX_SLOT][MAX_KERNEL];
	cl_event			voc_oclkrnwait[MAX_SLOT];
	cl_mem				voc_args[MAX_SLOT][MAX_ARG];
	size_t				voc_arg_size[MAX_SLOT][MAX_ARG];

	int				voc_pattern_rewrite;
	int				voc_pattern_alloc;

	vg_ocl_check_t			voc_verify_func[MAX_KERNEL];

	pthread_t			voc_ocl_thread;
	pthread_mutex_t			voc_lock;
	pthread_cond_t			voc_wait;
	int				voc_ocl_slot;
	int				voc_ocl_rows;
	int				voc_ocl_cols;
	int				voc_ocl_invsize;
	int				voc_halt;
	int				voc_dump_done;
	size_t				voc_ocl_bitmap_size;
};

extern char * save_file_name;

/* Thread synchronization stubs */
void
vg_exec_downgrade_lock(vg_exec_context_t *vxcp)
{
}

int
vg_exec_upgrade_lock(vg_exec_context_t *vxcp)
{
	return 0;
}


/*
 * OpenCL debugging and support
 */

const char *
vg_ocl_strerror(cl_int ret)
{
#define OCL_STATUS(st) case st: return #st;
	switch (ret) {
		OCL_STATUS(CL_SUCCESS);
		OCL_STATUS(CL_DEVICE_NOT_FOUND);
		OCL_STATUS(CL_DEVICE_NOT_AVAILABLE);
		OCL_STATUS(CL_COMPILER_NOT_AVAILABLE);
		OCL_STATUS(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		OCL_STATUS(CL_OUT_OF_RESOURCES);
		OCL_STATUS(CL_OUT_OF_HOST_MEMORY);
		OCL_STATUS(CL_PROFILING_INFO_NOT_AVAILABLE);
		OCL_STATUS(CL_MEM_COPY_OVERLAP);
		OCL_STATUS(CL_IMAGE_FORMAT_MISMATCH);
		OCL_STATUS(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		OCL_STATUS(CL_BUILD_PROGRAM_FAILURE);
		OCL_STATUS(CL_MAP_FAILURE);
#if defined(CL_MISALIGNED_SUB_BUFFER_OFFSET)
		OCL_STATUS(CL_MISALIGNED_SUB_BUFFER_OFFSET);
#endif /* defined(CL_MISALIGNED_SUB_BUFFER_OFFSET) */
#if defined(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		OCL_STATUS(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif /* defined(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) */
		OCL_STATUS(CL_INVALID_VALUE);
		OCL_STATUS(CL_INVALID_DEVICE_TYPE);
		OCL_STATUS(CL_INVALID_PLATFORM);
		OCL_STATUS(CL_INVALID_DEVICE);
		OCL_STATUS(CL_INVALID_CONTEXT);
		OCL_STATUS(CL_INVALID_QUEUE_PROPERTIES);
		OCL_STATUS(CL_INVALID_COMMAND_QUEUE);
		OCL_STATUS(CL_INVALID_HOST_PTR);
		OCL_STATUS(CL_INVALID_MEM_OBJECT);
		OCL_STATUS(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		OCL_STATUS(CL_INVALID_IMAGE_SIZE);
		OCL_STATUS(CL_INVALID_SAMPLER);
		OCL_STATUS(CL_INVALID_BINARY);
		OCL_STATUS(CL_INVALID_BUILD_OPTIONS);
		OCL_STATUS(CL_INVALID_PROGRAM);
		OCL_STATUS(CL_INVALID_PROGRAM_EXECUTABLE);
		OCL_STATUS(CL_INVALID_KERNEL_NAME);
		OCL_STATUS(CL_INVALID_KERNEL_DEFINITION);
		OCL_STATUS(CL_INVALID_KERNEL);
		OCL_STATUS(CL_INVALID_ARG_INDEX);
		OCL_STATUS(CL_INVALID_ARG_VALUE);
		OCL_STATUS(CL_INVALID_ARG_SIZE);
		OCL_STATUS(CL_INVALID_KERNEL_ARGS);
		OCL_STATUS(CL_INVALID_WORK_DIMENSION);
		OCL_STATUS(CL_INVALID_WORK_GROUP_SIZE);
		OCL_STATUS(CL_INVALID_WORK_ITEM_SIZE);
		OCL_STATUS(CL_INVALID_GLOBAL_OFFSET);
		OCL_STATUS(CL_INVALID_EVENT_WAIT_LIST);
		OCL_STATUS(CL_INVALID_EVENT);
		OCL_STATUS(CL_INVALID_OPERATION);
		OCL_STATUS(CL_INVALID_GL_OBJECT);
		OCL_STATUS(CL_INVALID_BUFFER_SIZE);
		OCL_STATUS(CL_INVALID_MIP_LEVEL);
		OCL_STATUS(CL_INVALID_GLOBAL_WORK_SIZE);
#if defined(CL_INVALID_PROPERTY)
		OCL_STATUS(CL_INVALID_PROPERTY);
#endif /* defined(CL_INVALID_PROPERTY) */
#undef OCL_STATUS
	default: {
		static char tmp[64];
		snprintf(tmp, sizeof(tmp), "Unknown code %d", ret);
		return tmp;
	}
	}
}

/* Get device strings, using a static buffer -- caveat emptor */
const char *
vg_ocl_platform_getstr(cl_platform_id pid, cl_platform_info param)
{
	static char platform_str[1024];
	cl_int ret;
	size_t size_ret;
	ret = clGetPlatformInfo(pid, param,
				sizeof(platform_str), platform_str,
				&size_ret);
	if (ret != CL_SUCCESS) {
		snprintf(platform_str, sizeof(platform_str),
			 "clGetPlatformInfo(%d): %s",
			 param, vg_ocl_strerror(ret));
	}
	return platform_str;
}

cl_platform_id
vg_ocl_device_getplatform(cl_device_id did)
{
	cl_int ret;
	cl_platform_id val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, CL_DEVICE_PLATFORM,
			      sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clGetDeviceInfo(CL_DEVICE_PLATFORM): %s",
			vg_ocl_strerror(ret));
	}
	return val;
}

cl_device_type
vg_ocl_device_gettype(cl_device_id did)
{
	cl_int ret;
	cl_device_type val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, CL_DEVICE_TYPE,
			      sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clGetDeviceInfo(CL_DEVICE_TYPE): %s",
			vg_ocl_strerror(ret));
	}
	return val;
}

const char *
vg_ocl_device_getstr(cl_device_id did, cl_device_info param)
{
	static char device_str[1024];
	cl_int ret;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param,
			      sizeof(device_str), device_str,
			      &size_ret);
	if (ret != CL_SUCCESS) {
		snprintf(device_str, sizeof(device_str),
			 "clGetDeviceInfo(%d): %s",
			 param, vg_ocl_strerror(ret));
	}
	return device_str;
}

size_t
vg_ocl_device_getsizet(cl_device_id did, cl_device_info param)
{
	cl_int ret;
	size_t val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param, sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr,
			"clGetDeviceInfo(%d): %s", param, vg_ocl_strerror(ret));
	}
	return val;
}

cl_ulong
vg_ocl_device_getulong(cl_device_id did, cl_device_info param)
{
	cl_int ret;
	cl_ulong val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param, sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr,
			"clGetDeviceInfo(%d): %s", param, vg_ocl_strerror(ret));
	}
	return val;
}

cl_uint
vg_ocl_device_getuint(cl_device_id did, cl_device_info param)
{
	cl_int ret;
        cl_uint val;
	size_t size_ret;
	ret = clGetDeviceInfo(did, param, sizeof(val), &val, &size_ret);
	if (ret != CL_SUCCESS) {
		fprintf(stderr,
			"clGetDeviceInfo(%d): %s", param, vg_ocl_strerror(ret));
	}
	return val;
}

void
vg_ocl_dump_info(vg_ocl_context_t *vocp)
{
	cl_device_id did;
	if (vocp->base.vxc_vc && (vocp->base.vxc_vc->vc_verbose < 1))
		return;
	if (vocp->voc_dump_done)
		return;
	did = vocp->voc_ocldid;
	fprintf(stderr, "Device: %s\n",
	       vg_ocl_device_getstr(did, CL_DEVICE_NAME));
	fprintf(stderr, "Vendor: %s (%04x)\n",
	       vg_ocl_device_getstr(did, CL_DEVICE_VENDOR),
	       vg_ocl_device_getuint(did, CL_DEVICE_VENDOR_ID));
	fprintf(stderr, "Driver: %s\n",
	       vg_ocl_device_getstr(did, CL_DRIVER_VERSION));
	fprintf(stderr, "Profile: %s\n",
	       vg_ocl_device_getstr(did, CL_DEVICE_PROFILE));
	fprintf(stderr, "Version: %s\n",
	       vg_ocl_device_getstr(did, CL_DEVICE_VERSION));
	fprintf(stderr, "Max compute units: %ld\n",
	       (long)vg_ocl_device_getsizet(did, CL_DEVICE_MAX_COMPUTE_UNITS));
	fprintf(stderr, "Max workgroup size: %ld\n",
	       (long)vg_ocl_device_getsizet(did, CL_DEVICE_MAX_WORK_GROUP_SIZE));
	fprintf(stderr, "Global memory: %lu\n",
	       vg_ocl_device_getulong(did, CL_DEVICE_GLOBAL_MEM_SIZE));
	fprintf(stderr, "Max allocation: %lu\n",
	       vg_ocl_device_getulong(did, CL_DEVICE_MAX_MEM_ALLOC_SIZE));
	vocp->voc_dump_done = 1;
}

void
vg_ocl_error(vg_ocl_context_t *vocp, int code, const char *desc)
{
	const char *err = vg_ocl_strerror(code);
	if (desc) {
		fprintf(stderr, "%s: %s\n", desc, err);
	} else {
		fprintf(stderr, "%s\n", err);
	}

	if (vocp && vocp->voc_ocldid)
		vg_ocl_dump_info(vocp);
}

void
vg_ocl_buildlog(vg_ocl_context_t *vocp, cl_program prog)
{
	size_t logbufsize, logsize;
	char *log;
	int off = 0;
	cl_int ret;

	ret = clGetProgramBuildInfo(prog,
				    vocp->voc_ocldid,
				    CL_PROGRAM_BUILD_LOG,
				    0, NULL,
				    &logbufsize);
	if (ret != CL_SUCCESS) {
		vg_ocl_error(NULL, ret, "clGetProgramBuildInfo");
		return;
	}

	log = (char *) malloc(logbufsize);
	if (!log) {
		fprintf(stderr, "Could not allocate build log buffer\n");
		return;
	}

	ret = clGetProgramBuildInfo(prog,
				    vocp->voc_ocldid,
				    CL_PROGRAM_BUILD_LOG,
				    logbufsize,
				    log,
				    &logsize);
	if (ret != CL_SUCCESS) {
		vg_ocl_error(NULL, ret, "clGetProgramBuildInfo");

	} else {
		/* Remove leading newlines and trailing newlines/whitespace */
		log[logbufsize-1] = '\0';
		for (off = logsize - 1; off >= 0; off--) {
			if ((log[off] != '\r') &&
			    (log[off] != '\n') &&
			    (log[off] != ' ') &&
			    (log[off] != '\t') &&
			    (log[off] != '\0'))
				break;
			log[off] = '\0';
		}
		for (off = 0; off < logbufsize; off++) {
			if ((log[off] != '\r') &&
			    (log[off] != '\n'))
				break;
		}

		fprintf(stderr, "Build log:\n%s\n", &log[off]);
	}
	free(log);
}

/*
 * OpenCL per-exec functions
 */

enum {
	VG_OCL_DEEP_PREPROC_UNROLL  = (1 << 0),
	VG_OCL_PRAGMA_UNROLL        = (1 << 1),
	VG_OCL_EXPENSIVE_BRANCHES   = (1 << 2),
	VG_OCL_DEEP_VLIW            = (1 << 3),
	VG_OCL_AMD_BFI_INT          = (1 << 4),
	VG_OCL_NV_VERBOSE           = (1 << 5),
	VG_OCL_BROKEN               = (1 << 6),
	VG_OCL_NO_BINARIES          = (1 << 7),

	VG_OCL_OPTIMIZATIONS        = (VG_OCL_DEEP_PREPROC_UNROLL |
				       VG_OCL_PRAGMA_UNROLL |
				       VG_OCL_EXPENSIVE_BRANCHES |
				       VG_OCL_DEEP_VLIW |
				       VG_OCL_AMD_BFI_INT),

};

int
vg_ocl_get_quirks(vg_ocl_context_t *vocp)
{
	uint32_t vend;
	const char *dvn;
	unsigned int quirks = 0;

	quirks |= VG_OCL_DEEP_PREPROC_UNROLL;

	vend = vg_ocl_device_getuint(vocp->voc_ocldid, CL_DEVICE_VENDOR_ID);
	switch (vend) {
	case 0x10de: /* NVIDIA */
		/*
		 * NVIDIA's compiler seems to take a really really long
		 * time when using preprocessor unrolling, but works
		 * well with pragma unroll.
		 */
		quirks &= ~VG_OCL_DEEP_PREPROC_UNROLL;
		quirks |= VG_OCL_PRAGMA_UNROLL;
		quirks |= VG_OCL_NV_VERBOSE;
		break;
	case 0x1002: /* AMD/ATI */
		/*
		 * AMD's compiler works best with preprocesor unrolling.
		 * Pragma unroll is unreliable with AMD's compiler and
		 * seems to crash based on whether the gods were smiling
		 * when Catalyst was last installed/upgraded.
		 */
		if (vg_ocl_device_gettype(vocp->voc_ocldid) &
		    CL_DEVICE_TYPE_GPU) {
			quirks |= VG_OCL_EXPENSIVE_BRANCHES;
			quirks |= VG_OCL_DEEP_VLIW;
			dvn = vg_ocl_device_getstr(vocp->voc_ocldid,
						   CL_DEVICE_EXTENSIONS);
			if (dvn && strstr(dvn, "cl_amd_media_ops"))
				quirks |= VG_OCL_AMD_BFI_INT;

			dvn = vg_ocl_device_getstr(vocp->voc_ocldid,
						   CL_DEVICE_NAME);
			if (!strcmp(dvn, "ATI RV710")) {
				quirks &= ~VG_OCL_OPTIMIZATIONS;
				quirks |= VG_OCL_NO_BINARIES;
			}
		}
		break;
	default:
		break;
	}
	return quirks;
}

int
vg_ocl_create_kernel(vg_ocl_context_t *vocp, int knum, const char *func)
{
	int i;
	cl_kernel krn;
	cl_int ret;

	for (i = 0; i < MAX_SLOT; i++) {
		krn = clCreateKernel(vocp->voc_oclprog, func, &ret);
		if (!krn) {
			fprintf(stderr, "clCreateKernel(%d): ", i);
			vg_ocl_error(vocp, ret, NULL);
			while (--i >= 0) {
				clReleaseKernel(vocp->voc_oclkernel[i][knum]);
				vocp->voc_oclkernel[i][knum] = NULL;
			}
			return 0;
		}
		
		vocp->voc_oclkernel[i][knum] = krn;
		vocp->voc_oclkrnwait[i] = NULL;
	}
	return 1;
}

void
vg_ocl_hash_program(vg_ocl_context_t *vocp, const char *opts,
		    const char *program, size_t size,
		    unsigned char *hash_out)
{
	EVP_MD_CTX *mdctx;
	cl_platform_id pid;
	const char *str;

	mdctx = EVP_MD_CTX_create();
	EVP_DigestInit_ex(mdctx, EVP_md5(), NULL);
	pid = vg_ocl_device_getplatform(vocp->voc_ocldid);
	str = vg_ocl_platform_getstr(pid, CL_PLATFORM_NAME);
	EVP_DigestUpdate(mdctx, str, strlen(str) + 1);
	str = vg_ocl_platform_getstr(pid, CL_PLATFORM_VERSION);
	EVP_DigestUpdate(mdctx, str, strlen(str) + 1);
	str = vg_ocl_device_getstr(vocp->voc_ocldid, CL_DEVICE_NAME);
	EVP_DigestUpdate(mdctx, str, strlen(str) + 1);
	if (opts)
		EVP_DigestUpdate(mdctx, opts, strlen(opts) + 1);
	if (size)
		EVP_DigestUpdate(mdctx, program, size);
	EVP_DigestFinal_ex(mdctx, hash_out, NULL);
	EVP_MD_CTX_destroy(mdctx);
}

typedef struct {
	unsigned char	e_ident[16];
	uint16_t	e_type;
	uint16_t	e_machine;
	uint32_t	e_version;
	uint32_t	e_entry;
	uint32_t	e_phoff;
	uint32_t	e_shoff;
	uint32_t	e_flags;
	uint16_t	e_ehsize;
	uint16_t	e_phentsize;
	uint16_t	e_phnum;
	uint16_t	e_shentsize;
	uint16_t	e_shnum;
	uint16_t	e_shstrndx;
} vg_elf32_header_t;

typedef struct {
	uint32_t	sh_name;
	uint32_t	sh_type;
	uint32_t	sh_flags;
	uint32_t	sh_addr;
	uint32_t	sh_offset;
	uint32_t	sh_size;
	uint32_t	sh_link;
	uint32_t	sh_info;
	uint32_t	sh_addralign;
	uint32_t	sh_entsize;
} vg_elf32_shdr_t;

int
vg_ocl_amd_patch_inner(unsigned char *binary, size_t size)
{
	vg_elf32_header_t *ehp;
	vg_elf32_shdr_t *shp, *nshp;
	uint32_t *instr;
	size_t off;
	int i, n, txt2idx, patched;

	ehp = (vg_elf32_header_t *) binary;
	if ((size < sizeof(*ehp)) ||
	    memcmp(ehp->e_ident, "\x7f" "ELF\1\1\1\x64", 8) ||
	    !ehp->e_shoff)
		return 0;

	off = ehp->e_shoff + (ehp->e_shstrndx * ehp->e_shentsize);
	nshp = (vg_elf32_shdr_t *) (binary + off);
	if ((off + sizeof(*nshp)) > size)
		return 0;

	shp = (vg_elf32_shdr_t *) (binary + ehp->e_shoff);
	n = 0;
	txt2idx = 0;
	for (i = 0; i < ehp->e_shnum; i++) {
		off = nshp->sh_offset + shp[i].sh_name;
		if (((off + 6) >= size) ||
		    memcmp(binary + off, ".text", 6))
			continue;
		n++;
		if (n == 2)
			txt2idx = i;
	}
	if (n != 2)
		return 0;

	off = shp[txt2idx].sh_offset;
	instr = (uint32_t *) (binary + off);
	n = shp[txt2idx].sh_size / 4;
	patched = 0;
	for (i = 0; i < n; i += 2) {
		if (((instr[i] & 0x02001000) == 0) &&
		    ((instr[i+1] & 0x9003f000) == 0x0001a000)) {
			instr[i+1] ^= (0x0001a000 ^ 0x0000c000);
			patched++;
		}
	}

	return patched;
}

int
vg_ocl_amd_patch(vg_ocl_context_t *vocp, unsigned char *binary, size_t size)
{
	vg_context_t *vcp = vocp->base.vxc_vc;
	vg_elf32_header_t *ehp;
	unsigned char *ptr;
	size_t offset = 1;
	int ninner = 0, nrun, npatched = 0;

	ehp = (vg_elf32_header_t *) binary;
	if ((size < sizeof(*ehp)) ||
	    memcmp(ehp->e_ident, "\x7f" "ELF\1\1\1\0", 8) ||
	    !ehp->e_shoff)
		return 0;

	offset = 1;
	while (offset < (size - 8)) {
		ptr = (unsigned char *) memchr(binary + offset,
					       0x7f,
					       size - offset);
		if (!ptr)
			return npatched;
		offset = ptr - binary;
		ehp = (vg_elf32_header_t *) ptr;
		if (((size - offset) < sizeof(*ehp)) ||
		    memcmp(ehp->e_ident, "\x7f" "ELF\1\1\1\x64", 8) ||
		    !ehp->e_shoff) {
			offset += 1;
			continue;
		}

		ninner++;
		nrun = vg_ocl_amd_patch_inner(ptr, size - offset);
		npatched += nrun;
		if (vcp->vc_verbose > 1)
			fprintf(stderr, "AMD BFI_INT: patched %d instructions "
			       "in kernel %d\n",
			       nrun, ninner);
		npatched++;
		offset += 1;
	}
	return npatched;
}


int
vg_ocl_load_program(vg_context_t *vcp, vg_ocl_context_t *vocp,
		    const char *filename, const char *opts)
{
	FILE *kfp;
	char *buf, *tbuf;
	int len, fromsource = 0, patched = 0;
	size_t sz, szr;
	cl_program prog;
	cl_int ret, sts;
	unsigned char prog_hash[16];
	char bin_name[64];

	if (vcp->vc_verbose > 1)
		fprintf(stderr,
			"OpenCL compiler flags: %s\n", opts ? opts : "");

	kfp = fopen(filename, "r");
	if (!kfp) {
		fprintf(stderr, "Error loading kernel file '%s': %s\n",
		       filename, strerror(errno));
		free(buf);
		return 0;
	}
    
    fseek(kfp, 0, SEEK_END);
    sz = ftell(kfp);
    fseek(kfp, 0, SEEK_SET);
	buf = (char *) malloc(sz+1);
	if (!buf) {
		fprintf(stderr, "Could not allocate program buffer\n");
		return 0;
	}

	len = fread(buf, 1, sz, kfp);
	fclose(kfp);
    buf[len] = 0;

	if (!len) {
		fprintf(stderr, "Short read on CL kernel\n");
		free(buf);
		return 0;
	}

	vg_ocl_hash_program(vocp, opts, buf, len, prog_hash);
	snprintf(bin_name, sizeof(bin_name),
		 "%02x%02x%02x%02x%02x%02x%02x%02x"
		 "%02x%02x%02x%02x%02x%02x%02x%02x.oclbin",
		 prog_hash[0], prog_hash[1], prog_hash[2], prog_hash[3],
		 prog_hash[4], prog_hash[5], prog_hash[6], prog_hash[7],
		 prog_hash[8], prog_hash[9], prog_hash[10], prog_hash[11],
		 prog_hash[12], prog_hash[13], prog_hash[14], prog_hash[15]);

	if (vocp->voc_quirks & VG_OCL_NO_BINARIES) {
		kfp = NULL;
		if (vcp->vc_verbose > 1)
			fprintf(stderr, "Binary OpenCL programs disabled\n");
	} else {
		kfp = fopen(bin_name, "rb");
	}

	if (!kfp) {
		/* No binary available, create with source */
		fromsource = 1;
		sz = len;
		prog = clCreateProgramWithSource(vocp->voc_oclctx,
						 1, (const char **) &buf, &sz,
						 &ret);
	} else {
		if (vcp->vc_verbose > 1)
			fprintf(stderr, "Loading kernel binary %s\n", bin_name);
		szr = 0;
		while (!feof(kfp)) {
			len = fread(buf + szr, 1, sz - szr, kfp);
			if (!len) {
				fprintf(stderr,
					"Short read on CL kernel binary\n");
				fclose(kfp);
				free(buf);
				return 0;
			}
			szr += len;
			if (szr == sz) {
				tbuf = (char *) realloc(buf, sz*2);
				if (!tbuf) {
					fprintf(stderr,
						"Could not expand CL kernel "
						"binary buffer\n");
					fclose(kfp);
					free(buf);
					return 0;
				}
				buf = tbuf;
				sz *= 2;
			}
		}
		fclose(kfp);
	rebuild:
		prog = clCreateProgramWithBinary(vocp->voc_oclctx,
						 1, &vocp->voc_ocldid,
						 &szr,
						 (const unsigned char **) &buf,
						 &sts,
						 &ret);
	}
	free(buf);
	if (!prog) {
		vg_ocl_error(vocp, ret, "clCreateProgramWithSource");
		return 0;
	}

	if (vcp->vc_verbose > 0) {
		if (fromsource && !patched) {
			fprintf(stderr,
				"Compiling kernel, can take minutes...");
			fflush(stderr);
		}
	}
	ret = clBuildProgram(prog, 1, &vocp->voc_ocldid, opts, NULL, NULL);
	if (ret != CL_SUCCESS) {
		if ((vcp->vc_verbose > 0) && fromsource && !patched)
			fprintf(stderr, "failure.\n");
		vg_ocl_error(NULL, ret, "clBuildProgram");
	} else if ((vcp->vc_verbose > 0) && fromsource && !patched) {
		fprintf(stderr, "done!\n");
	}
	if ((ret != CL_SUCCESS) ||
	    ((vcp->vc_verbose > 1) && fromsource && !patched)) {
		vg_ocl_buildlog(vocp, prog);
	}
	if (ret != CL_SUCCESS) {
		vg_ocl_dump_info(vocp);
		clReleaseProgram(prog);
		return 0;
	}

	if (fromsource && !(vocp->voc_quirks & VG_OCL_NO_BINARIES)) {
		ret = clGetProgramInfo(prog,
				       CL_PROGRAM_BINARY_SIZES,
				       sizeof(szr), &szr,
				       &sz);
		if (ret != CL_SUCCESS) {
			vg_ocl_error(vocp, ret,
				     "WARNING: clGetProgramInfo(BINARY_SIZES)");
			goto out;
		}
		if (sz == 0) {
			fprintf(stderr,
				"WARNING: zero-length CL kernel binary\n");
			goto out;
		}

		buf = (char *) malloc(szr);
		if (!buf) {
			fprintf(stderr,
				"WARNING: Could not allocate %ld bytes "
				"for CL binary\n",
			       (long)szr);
			goto out;
		}

		ret = clGetProgramInfo(prog,
				       CL_PROGRAM_BINARIES,
				       sizeof(buf), &buf,
				       &sz);
		if (ret != CL_SUCCESS) {
			vg_ocl_error(vocp, ret,
				     "WARNING: clGetProgramInfo(BINARIES)");
			free(buf);
			goto out;
		}

		if ((vocp->voc_quirks & VG_OCL_AMD_BFI_INT) && !patched) {
			patched = vg_ocl_amd_patch(vocp,
						   (unsigned char *) buf, szr);
			if (patched > 0) {
				if (vcp->vc_verbose > 1)
					fprintf(stderr,
						"AMD BFI_INT patch complete\n");
				clReleaseProgram(prog);
				goto rebuild;
			}
			fprintf(stderr,
				"WARNING: AMD BFI_INT patching failed\n");
			if (patched < 0) {
				/* Program was incompletely modified */
				free(buf);
				goto out;
			}
		}

		kfp = fopen(bin_name, "wb");
		if (!kfp) {
			fprintf(stderr, "WARNING: "
				"could not save CL kernel binary: %s\n",
				strerror(errno));
		} else {
			sz = fwrite(buf, 1, szr, kfp);
			fclose(kfp);
			if (sz != szr) {
				fprintf(stderr,
					"WARNING: short write on CL kernel "
					"binary file: expected "
					"%ld, got %ld\n",
					(long)szr, (long)sz);
				unlink(bin_name);
			}
		}
		free(buf);
	}

out:
	vocp->voc_oclprog = prog;
	if (!vg_ocl_create_kernel(vocp, 0, "ec_generate_points") ||
	    !vg_ocl_create_kernel(vocp, 1, "heap_invert_and_check") ||
	    !vg_ocl_create_kernel(vocp, 2, "fill_bitmap") ||
	    !vg_ocl_create_kernel(vocp, 3, "ec_add_grid")
	    ) {
		clReleaseProgram(vocp->voc_oclprog);
		vocp->voc_oclprog = NULL;
		return 0;
	}

	return 1;
}

void CL_CALLBACK
vg_ocl_context_callback(const char *errinfo,
			const void *private_info,
			size_t cb,
			void *user_data)
{
	fprintf(stderr, "vg_ocl_context_callback error: %s\n", errinfo);
}

int
vg_ocl_init(vg_context_t *vcp, vg_ocl_context_t *vocp, cl_device_id did,
	    int safe_mode)
{
	cl_int ret;
	char optbuf[128];
	int end = 0;

	memset(vocp, 0, sizeof(*vocp));
	vg_exec_context_init(vcp, &vocp->base);
	vocp->base.vxc_threadfunc = vg_opencl_loop;

	pthread_mutex_init(&vocp->voc_lock, NULL);
	pthread_cond_init(&vocp->voc_wait, NULL);
	vocp->voc_ocl_slot = -1;

	vocp->voc_ocldid = did;

	cl_ulong localmemsize = vg_ocl_device_getulong(vocp->voc_ocldid, CL_DEVICE_LOCAL_MEM_SIZE);
	
	if (vcp->vc_verbose > 1)
		vg_ocl_dump_info(vocp);

	vocp->voc_quirks = vg_ocl_get_quirks(vocp);

	if ((vocp->voc_quirks & VG_OCL_BROKEN) && (vcp->vc_verbose > 0)) {
		char yesbuf[16];
		printf("Type 'yes' to continue: ");
		fflush(stdout);
		if (!fgets(yesbuf, sizeof(yesbuf), stdin) ||
		    strncmp(yesbuf, "yes", 3))
			exit(1);
	}

	vocp->voc_oclctx = clCreateContext(NULL,
					   1, &did,
					   vg_ocl_context_callback,
					   NULL,
					   &ret);
	if (!vocp->voc_oclctx) {
		vg_ocl_error(vocp, ret, "clCreateContext");
		return 0;
	}

	vocp->voc_oclcmdq = clCreateCommandQueue(vocp->voc_oclctx,
						 vocp->voc_ocldid,
						 0, &ret);
	if (!vocp->voc_oclcmdq) {
		vg_ocl_error(vocp, ret, "clCreateCommandQueue");
		return 0;
	}

	if (safe_mode)
		vocp->voc_quirks &= ~VG_OCL_OPTIMIZATIONS;

	end = 0;
	optbuf[end] = '\0';
	if (vocp->voc_quirks & VG_OCL_DEEP_PREPROC_UNROLL)
		end += snprintf(optbuf + end, sizeof(optbuf) - end,
				"-DDEEP_PREPROC_UNROLL ");
	if (vocp->voc_quirks & VG_OCL_PRAGMA_UNROLL)
		end += snprintf(optbuf + end, sizeof(optbuf) - end,
				"-DPRAGMA_UNROLL ");
	if (vocp->voc_quirks & VG_OCL_EXPENSIVE_BRANCHES)
		end += snprintf(optbuf + end, sizeof(optbuf) - end,
				"-DVERY_EXPENSIVE_BRANCHES ");
	if (vocp->voc_quirks & VG_OCL_DEEP_VLIW)
		end += snprintf(optbuf + end, sizeof(optbuf) - end,
				"-DDEEP_VLIW ");
	if (vocp->voc_quirks & VG_OCL_AMD_BFI_INT)
		end += snprintf(optbuf + end, sizeof(optbuf) - end,
				"-DAMD_BFI_INT ");
	if (vocp->voc_quirks & VG_OCL_NV_VERBOSE)
		end += snprintf(optbuf + end, sizeof(optbuf) - end,
				"-cl-nv-verbose -cl-nv-maxrregcount=85 ");
	end += snprintf(optbuf + end, sizeof(optbuf) - end, "-DLOCAL_MEM_SIZE=%ld ", localmemsize-16);

	if (!vg_ocl_load_program(vcp, vocp, "calc_addrs.cl", optbuf))
		return 0;
	return 1;
}

void
vg_ocl_del(vg_ocl_context_t *vocp)
{
	vg_ocl_free_args(vocp);
	if (vocp->voc_oclprog) {
		clReleaseProgram(vocp->voc_oclprog);
		vocp->voc_oclprog = NULL;
	}
	if (vocp->voc_oclcmdq) {
		clReleaseCommandQueue(vocp->voc_oclcmdq);
		vocp->voc_oclcmdq = NULL;
	}
	if (vocp->voc_oclctx) {
		clReleaseContext(vocp->voc_oclctx);
		vocp->voc_oclctx = NULL;
	}
	pthread_cond_destroy(&vocp->voc_wait);
	pthread_mutex_destroy(&vocp->voc_lock);
	vg_exec_context_del(&vocp->base);
}

enum Arg {
	A_FOUND = 0,
	A_Z_HEAP,
	A_POINTS,
	A_INCREMENT,
    A_START,
	A_ADDRESSES,
	A_ADDRESS_COUNT,
	A_BITMAP,
	A_BITMAP_LEN,
};

int vg_ocl_arg_map[][8] = {
	/* found */
	{ 1, 5, -1 },
	/* z_heap */
	{ 0, 1, 1, 0, 3, 1, -1 },
	/* points */
	{ 0, 0, 1, 2, 3, 0, -1 },
	/* increment */
	{ 3, 2, -1 },
    /* start */
    { 0, 2, -1},
	/* addresses */
	{ 2, 0, 1, 6, -1},
	/* address count */
	{ 2, 1, 1, 7, -1},
	/* bitmap */
	{ 1, 3, 2,2, -1 },
	/* bitmap_len */
	{ 1, 4, 2,3, -1 },
};

cl_mem
vg_ocl_kernel_arg_alloc(vg_ocl_context_t *vocp, int slot,
			int arg, size_t size, int host)
{
	cl_mem clbuf;
	cl_int ret;
	int i, j, knum, karg;

	for (i = 0; i < MAX_SLOT; i++) {
		if ((i != slot) && (slot >= 0))
			continue;
		if (vocp->voc_args[i][arg]) {
			clReleaseMemObject(vocp->voc_args[i][arg]);
			vocp->voc_args[i][arg] = NULL;
			vocp->voc_arg_size[i][arg] = 0;
		}
	}

	clbuf = clCreateBuffer(vocp->voc_oclctx,
			       CL_MEM_READ_WRITE |
			       (host ? CL_MEM_ALLOC_HOST_PTR : 0),
			       size,
			       NULL,
			       &ret);
	if (!clbuf) {
		fprintf(stderr, "clCreateBuffer(%d,%d) size=%ld: ", slot, arg, size);
		vg_ocl_error(vocp, ret, NULL);
		return 0;
	}

	for (i = 0; i < MAX_SLOT; i++) {
		if ((i != slot) && (slot >= 0))
			continue;

		clRetainMemObject(clbuf);
		vocp->voc_args[i][arg] = clbuf;
		vocp->voc_arg_size[i][arg] = size;

		for (j = 0; vg_ocl_arg_map[arg][j] >= 0; j += 2) {
			knum = vg_ocl_arg_map[arg][j];
			karg = vg_ocl_arg_map[arg][j+1];
			ret = clSetKernelArg(vocp->voc_oclkernel[i][knum],
					     karg,
					     sizeof(clbuf),
					     &clbuf);
			
			if (ret) {
				fprintf(stderr,
					"cl_mem clSetKernelArg(kernel=%d,karg=%d,arg=%d) slot=%d: ", knum, karg, arg, i);
				vg_ocl_error(vocp, ret, NULL);
				return 0;
			}
		}
	}

	clReleaseMemObject(clbuf);
	return clbuf;
}

int
vg_ocl_copyout_arg(vg_ocl_context_t *vocp, int wslot, int arg,
		   void *buffer, size_t size)
{
	cl_int slot, ret;

	slot = (wslot < 0) ? 0 : wslot;

	assert((slot >= 0) && (slot < MAX_SLOT));
	assert(size <= vocp->voc_arg_size[slot][arg]);

	ret = clEnqueueWriteBuffer(vocp->voc_oclcmdq,
				   vocp->voc_args[slot][arg],
				   CL_TRUE,
				   0, size,
				   buffer,
				   0, NULL,
				   NULL);
			
	if (ret) {
		fprintf(stderr, "clEnqueueWriteBuffer(%d): ", arg);
		vg_ocl_error(vocp, ret, NULL);
		return 0;
	}

	return 1;
}

void *
vg_ocl_map_arg_buffer(vg_ocl_context_t *vocp, int slot,
		      int arg, int rw)
{
	void *buf;
	cl_int ret;

	assert((slot >= 0) && (slot < MAX_SLOT));

	buf = clEnqueueMapBuffer(vocp->voc_oclcmdq,
				 vocp->voc_args[slot][arg],
				 CL_TRUE,
				 (rw == 2) ? (CL_MAP_READ|CL_MAP_WRITE)
				           : (rw ? CL_MAP_WRITE : CL_MAP_READ),
				 0, vocp->voc_arg_size[slot][arg],
				 0, NULL,
				 NULL,
				 &ret);
	if (!buf) {
		fprintf(stderr, "clEnqueueMapBuffer(%d): ", arg);
		vg_ocl_error(vocp, ret, NULL);
		return NULL;
	}
	return buf;
}

void
vg_ocl_read_arg_buffer(vg_ocl_context_t *vocp, int slot,
		      int arg, void * mem)
{
	cl_int ret;

	assert((slot >= 0) && (slot < MAX_SLOT));

	ret = clEnqueueReadBuffer(vocp->voc_oclcmdq,
				 vocp->voc_args[slot][arg],
				 CL_TRUE,
				 0, vocp->voc_arg_size[slot][arg],
				 mem, 0, NULL,NULL);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueReadBuffer(%d): ", arg);
		vg_ocl_error(vocp, ret, NULL);
		return ;
	}
}

void
vg_ocl_unmap_arg_buffer(vg_ocl_context_t *vocp, int slot,
			int arg, void *buf)
{
	cl_int ret;
	cl_event ev;

	assert((slot >= 0) && (slot < MAX_SLOT));

	ret = clEnqueueUnmapMemObject(vocp->voc_oclcmdq,
				      vocp->voc_args[slot][arg],
				      buf,
				      0, NULL,
				      &ev);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueUnmapMemObject(%d): ", arg);
		vg_ocl_error(vocp, ret, NULL);
		return;
	}

	ret = clWaitForEvents(1, &ev);
	clReleaseEvent(ev);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clWaitForEvent(clUnmapMemObject,%d): ", arg);
		vg_ocl_error(vocp, ret, NULL);
	}
}

void
vg_ocl_write_arg_buffer(vg_ocl_context_t *vocp, int slot, int arg, void *buf)
{
	cl_int ret;

	assert((slot >= 0) && (slot < MAX_SLOT));

	ret = clEnqueueWriteBuffer(vocp->voc_oclcmdq,
				      vocp->voc_args[slot][arg],
                      CL_FALSE,
                      0, vocp->voc_arg_size[slot][arg],
				      buf,
				      0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer(%d): ", arg);
		vg_ocl_error(vocp, ret, NULL);
		return;
	}
}

int
vg_ocl_kernel_int_arg(vg_ocl_context_t *vocp, int slot,
		      int arg, int value)
{
	cl_int ret;
	int i, j, knum, karg;

	for (i = 0; i < MAX_SLOT; i++) {
		if ((i != slot) && (slot >= 0))
			continue;
		for (j = 0; vg_ocl_arg_map[arg][j] >= 0; j += 2) {
			knum = vg_ocl_arg_map[arg][j];
			karg = vg_ocl_arg_map[arg][j+1];
			ret = clSetKernelArg(vocp->voc_oclkernel[i][knum],
					     karg,
					     sizeof(value),
					     &value);
			if (ret) {
				fprintf(stderr, "clSetKernelArg(%d): ", arg);
				vg_ocl_error(vocp, ret, NULL);
				return 0;
			}
		}
	}
	return 1;
}

int
vg_ocl_kernel_long_arg(vg_ocl_context_t *vocp, int slot,
		      int arg, cl_long value)
{
	cl_int ret;
	int i, j, knum, karg;

	for (i = 0; i < MAX_SLOT; i++) {
		if ((i != slot) && (slot >= 0))
			continue;
		for (j = 0; vg_ocl_arg_map[arg][j] >= 0; j += 2) {
			knum = vg_ocl_arg_map[arg][j];
			karg = vg_ocl_arg_map[arg][j+1];
			ret = clSetKernelArg(vocp->voc_oclkernel[i][knum],
					     karg,
					     sizeof(value),
					     &value);
			if (ret) {
				fprintf(stderr, "clSetKernelArg(%d): ", arg);
				vg_ocl_error(vocp, ret, NULL);
				return 0;
			}
		}
	}
	return 1;
}

int
vg_ocl_kernel_buffer_arg(vg_ocl_context_t *vocp, int slot,
			 int arg, void *value, size_t size)
{
	cl_int ret;
	int i, j, knum, karg;

	for (i = 0; i < MAX_SLOT; i++) {
		if ((i != slot) && (slot >= 0))
			continue;
		for (j = 0; vg_ocl_arg_map[arg][j] >= 0; j += 2) {
			knum = vg_ocl_arg_map[arg][j];
			karg = vg_ocl_arg_map[arg][j+1];
			ret = clSetKernelArg(vocp->voc_oclkernel[i][knum],
					     karg,
					     size,
					     value);
			if (ret) {
				fprintf(stderr,
					"vg_ocl_kernel_buffer_arg clSetKernelArg(%d,%d): ", knum, karg);
				vg_ocl_error(vocp, ret, NULL);
				return 0;
			}
		}
	}
	return 1;
}

void
vg_ocl_free_args(vg_ocl_context_t *vocp)
{
	int i, arg;
	for (i = 0; i < MAX_SLOT; i++) {
		for (arg = 0; arg < MAX_ARG; arg++) {
			if (vocp->voc_args[i][arg]) {
				clReleaseMemObject(vocp->voc_args[i][arg]);
				vocp->voc_args[i][arg] = NULL;
				vocp->voc_arg_size[i][arg] = 0;
			}
		}
	}
}

int
vg_ocl_kernel_dead(vg_ocl_context_t *vocp, int slot)
{
	return (vocp->voc_oclkrnwait[slot] == NULL);
}

int
vg_ocl_kernel_init(vg_ocl_context_t *vocp, int npoints, BIGNUM * start)
{
	cl_int ret;
	cl_event ev;
	size_t globalws[1] = { npoints };

    // set the 'start' bignum argument
    unsigned char bytes[32];
    memset(bytes, 0, 32);
    BN_bn2bin(start, bytes+32-BN_num_bytes(start));
    for (int i=0; i<16; i++) {
        unsigned tmp = bytes[i];
        bytes[i] = bytes[31-i];
        bytes[31-i] = tmp;
    }
    vg_ocl_write_arg_buffer(vocp, 0, A_START, bytes);
    
	ret = clEnqueueNDRangeKernel(vocp->voc_oclcmdq,
				     vocp->voc_oclkernel[0][0],
				     1,
				     NULL, globalws, NULL,
				     0, NULL,
				     &ev);
	if (ret != CL_SUCCESS) {
		vg_ocl_error(vocp, ret, "clEnqueueNDRange(0)");
		return 0;
	}
    return 1;
}

int
vg_ocl_kernel_start(vg_ocl_context_t *vocp, int slot, int npoints, int invsize, BIGNUM * start, int first)
{
	cl_int val, ret;
	cl_event ev;
	size_t globalws[1] = { npoints };
	size_t invws = npoints / invsize;

	assert(!vocp->voc_oclkrnwait[slot]);

	/* heap_invert() preconditions */
	assert(is_pow2(invsize) && (invsize > 1));

	val = invsize;
    // batch argument for heap_invert
	ret = clSetKernelArg(vocp->voc_oclkernel[slot][1],
			     1,
			     sizeof(val),
			     &val);
	if (ret != CL_SUCCESS) {
		vg_ocl_error(vocp, ret, "clSetKernelArg(ncol)");
		return 0;
	}
    
    if (first) {
        // set the 'start' bignum argument
        unsigned char bytes[32];
        memset(bytes, 0, 32);
        BN_bn2bin(start, bytes+32-BN_num_bytes(start));
        for (int i=0; i<16; i++) {
            unsigned tmp = bytes[i];
            bytes[i] = bytes[31-i];
            bytes[31-i] = tmp;
        }
        vg_ocl_write_arg_buffer(vocp, 0, A_START, bytes);
        
        // ec_generate_points
        ret = clEnqueueNDRangeKernel(vocp->voc_oclcmdq,
                         vocp->voc_oclkernel[0][0],
                         1,
                         NULL, globalws, NULL,
                         0, NULL,
                         &ev);
    } else {
        // ec_add_grid
        ret = clEnqueueNDRangeKernel(vocp->voc_oclcmdq,
				     vocp->voc_oclkernel[slot][3],
				     1,
				     NULL, globalws, NULL,
				     0, NULL,
				     &ev);
    }
	if (ret != CL_SUCCESS) {
		vg_ocl_error(vocp, ret, "clEnqueueNDRange");
		return 0;
	}

	ret = clWaitForEvents(1, &ev);
	clReleaseEvent(ev);
	if (ret != CL_SUCCESS) {
		vg_ocl_error(vocp, ret, "clWaitForEvents(NDRange,0)");
		return 0;
	}

	if (vocp->voc_verify_func[0] &&
	    !(vocp->voc_verify_func[0])(vocp, slot)) {
		fprintf(stderr, "ERROR: Kernel 0 failed verification test\n");
		return 0;
	}
       
        // heap_invert_and_check(z_heap,batch,points_in,bitmap, bitmap_len, found)
	ret = clEnqueueNDRangeKernel(vocp->voc_oclcmdq,
				     vocp->voc_oclkernel[slot][1],
				     1,
				     NULL, &invws, NULL,
				     0, NULL,
				     &ev);
	if (ret != CL_SUCCESS) {
		fprintf(stderr, "slot=%d\n", slot);
		vg_ocl_error(vocp, ret, "clEnqueueNDRange(1)");
		return 0;
	}

	vocp->voc_oclkrnwait[slot] = ev;
	return 1;
}

int
vg_ocl_kernel_wait(vg_ocl_context_t *vocp, int slot)
{
	cl_event ev;
	cl_int ret;

	ev = vocp->voc_oclkrnwait[slot];
	vocp->voc_oclkrnwait[slot] = NULL;
	if (ev) {
		ret = clWaitForEvents(1, &ev);
		clReleaseEvent(ev);
		if (ret != CL_SUCCESS) {
			vg_ocl_error(vocp, ret, "clWaitForEvents(NDRange,e)");
			return 0;
		}
	}
	return 1;
}


void vg_ocl_get_bignum_raw(BIGNUM *bn, const unsigned char *buf)
{
	bn_expand(bn, 256);
	memcpy(bn->d, buf, 32);
	bn->top = (32 / sizeof(BN_ULONG));
}

void vg_ocl_put_bignum_raw(unsigned char *buf, const BIGNUM *bn)
{
	int bnlen = (bn->top * sizeof(BN_ULONG));
	if (bnlen >= 32) {
		memcpy(buf, bn->d, 32);
	} else {
		memcpy(buf, bn->d, bnlen);
		memset(buf + bnlen, 0, 32 - bnlen);
	}
}

#define ACCESS_BUNDLE 1024
#define ACCESS_STRIDE (ACCESS_BUNDLE/8)

void
vg_ocl_get_bignum_tpa(BIGNUM *bn, const unsigned char *buf, int cell)
{
	unsigned char bnbuf[32];
	int start, i;

	start = (((cell / ACCESS_STRIDE) * ACCESS_BUNDLE) +
		 (cell % ACCESS_STRIDE));
	for (i = 0; i < 8; i++)
		memcpy(bnbuf+(i*4),
		       buf + 4*(start + i*ACCESS_STRIDE),
		       4);

	vg_ocl_get_bignum_raw(bn, bnbuf);
}

/*
 * Absolutely disgusting.
 * We want points in Montgomery form, and it's a lot easier to read the
 * coordinates from the structure than to export and re-montgomeryize.
 */

struct ec_point_st {
	const EC_METHOD *meth;
	BIGNUM X;
	BIGNUM Y;
	BIGNUM Z;
	int Z_is_one;
};

void vg_ocl_get_point(EC_POINT *ppnt, const unsigned char *buf)
{
	static const unsigned char mont_one[] = { 0x01,0x00,0x00,0x03,0xd1 };
	vg_ocl_get_bignum_raw(&ppnt->X, buf);
	vg_ocl_get_bignum_raw(&ppnt->Y, buf + 32);
	if (!ppnt->Z_is_one) {
		ppnt->Z_is_one = 1;
		BN_bin2bn(mont_one, sizeof(mont_one), &ppnt->Z);
	}
}

void vg_ocl_put_point(unsigned char *buf, const EC_POINT *ppnt)
{
	assert(ppnt->Z_is_one);
	vg_ocl_put_bignum_raw(buf, &ppnt->X);
	vg_ocl_put_bignum_raw(buf + 32, &ppnt->Y);
}

void vg_ocl_put_point_tpa(unsigned char *buf, int cell, const EC_POINT *ppnt)
{
	unsigned char pntbuf[64];
	int start, i;

	vg_ocl_put_point(pntbuf, ppnt);

	start = ((((2 * cell) / ACCESS_STRIDE) * ACCESS_BUNDLE) +
		 (cell % (ACCESS_STRIDE/2)));
	for (i = 0; i < 8; i++)
		memcpy(buf + 4*(start + i*ACCESS_STRIDE),
		       pntbuf+(i*4),
		       4);
	for (i = 0; i < 8; i++)
		memcpy(buf + 4*(start + (ACCESS_STRIDE/2) + (i*ACCESS_STRIDE)),
		       pntbuf+32+(i*4),
		       4);
}

void vg_ocl_get_point_tpa(EC_POINT *ppnt, const unsigned char *buf, int cell)
{
	unsigned char pntbuf[64];
	int start, i;

	start = ((((2 * cell) / ACCESS_STRIDE) * ACCESS_BUNDLE) +
		 (cell % (ACCESS_STRIDE/2)));
	for (i = 0; i < 8; i++)
		memcpy(pntbuf+(i*4),
		       buf + 4*(start + i*ACCESS_STRIDE),
		       4);
	for (i = 0; i < 8; i++)
		memcpy(pntbuf+32+(i*4),
		       buf + 4*(start + (ACCESS_STRIDE/2) + (i*ACCESS_STRIDE)),
		       4);

	vg_ocl_get_point(ppnt, pntbuf);
}

void
show_elapsed(struct timeval *tv, const char *place)
{
	struct timeval now, delta;
        gettimeofday(&now, NULL);
	timersub(&now, tv, &delta);
	fprintf(stderr,
		"%s spent %ld.%06lds\n", place, delta.tv_sec, delta.tv_usec);
}


int vg_ocl_gethash_check(vg_ocl_context_t *vocp, int slot)
{
    return 1;
}

int vg_ocl_gethash_init(vg_ocl_context_t *vocp)
{
	return 1;
}

#define OCL_CHECK(result) {cl_int ret = result; \
	if (ret!=CL_SUCCESS) { fprintf(stderr, "file: %s line: %d\n", __FILE__, __LINE__); vg_ocl_error(vocp, ret, NULL);}\
	}

int vg_ocl_prefix_rekey(vg_ocl_context_t *vocp)
{
	vg_context_t *vcp = vocp->base.vxc_vc;
	unsigned char *ocl_targets_in;
	uint32_t *ocl_found_out;
	int i;
	
	/* Set the found indicator for each slot to -1 */
	for (i = 0; i < vocp->voc_nslots; i++) {
		ocl_found_out = (uint32_t *)
			vg_ocl_map_arg_buffer(vocp, i, A_FOUND, 1);
		if (!ocl_found_out) {
			fprintf(stderr,
				"ERROR: Could not map result buffer"
				" for slot %d (rekey)\n", i);
			return -1;
		}
		ocl_found_out[0] = 0xffffffff;
		vg_ocl_unmap_arg_buffer(vocp, i, A_FOUND, ocl_found_out);
	}

	if (vocp->voc_pattern_rewrite) {
                /* (re)allocate target buffer */
                if (!vg_ocl_kernel_arg_alloc(vocp, -1, A_ADDRESSES, 20 * vcp->vc_npatterns, 0))
                        return -1;
                
                /* (re)allocate bitmap buffer */
                if (!vg_ocl_kernel_arg_alloc(vocp, -1, A_BITMAP, vocp->voc_ocl_bitmap_size/8, 0))
                        return -1;
		
                vocp->voc_pattern_alloc = i;

		/* Write patterns */
		ocl_targets_in = (unsigned char *)vg_ocl_map_arg_buffer(vocp, 0, A_ADDRESSES, 1);
		if (!ocl_targets_in) {
			fprintf(stderr,
				"ERROR: Could not map hash target buffer\n");
			return -1;
		}
                memcpy(ocl_targets_in, vcp->patterns, vcp->vc_npatterns*20);
		vg_ocl_unmap_arg_buffer(vocp, 0, A_ADDRESSES, ocl_targets_in);
		vg_ocl_kernel_int_arg(vocp, -1, A_ADDRESS_COUNT, vcp->vc_npatterns);
		vg_ocl_kernel_long_arg(vocp, -1, A_BITMAP_LEN, vocp->voc_ocl_bitmap_size);

		// fill_bitmap
		size_t globalws[1] = {1024};
		OCL_CHECK(clEnqueueNDRangeKernel(vocp->voc_oclcmdq,
				     vocp->voc_oclkernel[0][2],
				     1,
				     NULL, globalws, NULL,
				     0, NULL,
				     NULL));

		vocp->voc_pattern_rewrite = 0;
	}
	return 1;
}

int vg_ocl_prefix_check(vg_ocl_context_t *vocp, int slot, BIGNUM * pkstart)
{
	vg_exec_context_t *vxcp = &vocp->base;
	vg_context_t *vcp = vocp->base.vxc_vc;
	uint32_t *ocl_found_out;
	int found_count;
	int res = 0;

	/* Retrieve the found indicator */
    
	ocl_found_out = (uint32_t *)malloc(vocp->voc_arg_size[slot][A_FOUND]);
    vg_ocl_read_arg_buffer(vocp, slot, A_FOUND, ocl_found_out);
	found_count = ocl_found_out[0];
	ocl_found_out[0] = -1;
    vg_ocl_write_arg_buffer(vocp, slot, A_FOUND, ocl_found_out);

	if (found_count != -1) {
        EC_KEY * save_key = vxcp->vxc_key;
        vxcp->vxc_key = EC_KEY_dup(vxcp->vxc_key);
		found_count = -found_count-1;
		if (vocp->base.vxc_vc->vc_verbose > 1) printf("\nFound %d candidadtes, check on CPU\n", found_count);
		/* GPU code claims match, verify with CPU version */
        res = 0;
		for (int i=0; i<found_count; i++) {
			BIGNUM * offset = BN_new();
            BN_set_word(offset, ocl_found_out[1+i]);
            BN_add(offset, offset, pkstart);
            vg_set_privkey(offset, vxcp->vxc_key);
			for (vxcp->vc_combined_compressed=0; vxcp->vc_combined_compressed<2; vxcp->vc_combined_compressed++) {
				if (vocp->base.vxc_vc->vc_verbose > 1) {printf("check key=");dumpbn(EC_KEY_get0_private_key(vxcp->vxc_key));}
                
                const EC_POINT *pubkey = EC_KEY_get0_public_key(vxcp->vxc_key);
                if (!pubkey) printf("pubkey is NULL!\n");
                unsigned char eckey_buf[96], hash1[32], hash2[20];
                int len;

                len = EC_POINT_point2oct(pgroup,
                             pubkey,
                             vxcp->vc_combined_compressed ? POINT_CONVERSION_COMPRESSED : POINT_CONVERSION_UNCOMPRESSED,
                             eckey_buf,
                             sizeof(eckey_buf),
                             vxcp->vxc_bnctx);
                                
                SHA256(eckey_buf, len, hash1);
                RIPEMD160(hash1, sizeof(hash1), hash2);
                memcpy(&vxcp->vxc_binres[1], hash2, 20);

				/* Make sure the GPU produced the expected hash */
				if (vg_prefix_test(vxcp)) {
/*
					char privkey_buf[VG_PROTKEY_MAX_B58];
					char addr_buf[64];
					EC_POINT * ppnt = (EC_POINT *) EC_KEY_get0_public_key(vxcp->vxc_key);
					if (vxcp->vc_combined_compressed) {
						vg_encode_address_compressed(ppnt,
							EC_KEY_get0_group(vxcp->vxc_key),
							vcp->vc_pubkeytype, addr_buf);
						vg_encode_privkey_compressed(vxcp->vxc_key, vcp->vc_privtype, privkey_buf);
					} else {
						vg_encode_address(ppnt,
							EC_KEY_get0_group(vxcp->vxc_key),
							vcp->vc_pubkeytype, addr_buf);
						vg_encode_privkey(vxcp->vxc_key, vcp->vc_privtype, privkey_buf);
					}
					//printf("Address: %s\nPrivkey: %s\n", addr_buf, privkey_buf);
                    */
					res = 1;
				}
			}
		}
		if (!res) if (vocp->base.vxc_vc->vc_verbose > 1) printf("CPU check failed :(\n");
        EC_KEY_free(vxcp->vxc_key);
        vxcp->vxc_key = save_key;
	}

    free(ocl_found_out);

	return res;
}

int vg_ocl_prefix_init(vg_ocl_context_t *vocp)
{
	if (!vg_ocl_create_kernel(vocp, 2, "fill_bitmap"))
		return 0;

	if (!vg_ocl_kernel_arg_alloc(vocp, -1, A_FOUND, 2049, 1))
		return 0;
	if (!vg_ocl_kernel_arg_alloc(vocp, -1, A_START, 32, 1))
		return 0;
	vocp->voc_rekey_func = vg_ocl_prefix_rekey;
	vocp->voc_pattern_rewrite = 1;
	vocp->voc_pattern_alloc = 0;
	return 1;
}


#define LIMBS(d) get_word(d,0), get_word(d,1), get_word(d,2), get_word(d,3), get_word(d,4), get_word(d,5), get_word(d,6), get_word(d,7)

unsigned get_word(BIGNUM * x, int i) {
    
    BIGNUM * y = BN_new();
    BN_rshift(y, x, i*32);
    BN_mask_bits(y, 32);
    unsigned w = BN_get_word(y);
    BN_free(y);
    return w;
}

/*
 * Address search thread main loop
 */
void *
vg_opencl_loop(vg_exec_context_t *arg)
{
	vg_ocl_context_t *vocp = (vg_ocl_context_t *) arg;
	int halt = 0, slot = 0;
	int round;

	EC_KEY *pkey = NULL;
	const EC_POINT *pgen;
	EC_POINT *poffset = NULL;

	vg_context_t *vcp = vocp->base.vxc_vc;
	vg_exec_context_t *vxcp = &vocp->base;

	struct timeval tvstart, tv;

	pkey = vxcp->vxc_key;
	pgroup = EC_KEY_get0_group(pkey);
	pgen = EC_GROUP_get0_generator(pgroup);

	round = vocp->voc_ocl_rows * vocp->voc_ocl_cols;

	vocp->voc_nslots = 1;

	if (!vg_ocl_prefix_init(vocp))
		return NULL;
	
	if (!vg_ocl_kernel_arg_alloc(vocp, -1, A_Z_HEAP,
			     round_up_pow2(32 * 2 * round, 4096), 0) ||
	    !vg_ocl_kernel_arg_alloc(vocp, -1, A_POINTS,
			     round_up_pow2(32 * 2 * round, 4096), 0) ||
	    !vg_ocl_kernel_arg_alloc(vocp, -1, A_INCREMENT, 32 * 2, 1) ||
	    !vg_ocl_kernel_arg_alloc(vocp, -1, A_START, 32, 1)
                 )
		return NULL;

	poffset = EC_POINT_new(pgroup);
	BN_set_word(&vxcp->vxc_bntmp, round);
    EC_POINT * igenerator = EC_POINT_dup(pgen, pgroup);
    EC_POINT_invert(pgroup, igenerator, vxcp->vxc_bnctx);
	EC_POINT_mul(pgroup, poffset, NULL, igenerator, &vxcp->vxc_bntmp, vxcp->vxc_bnctx);
	EC_POINT_make_affine(pgroup, poffset, vxcp->vxc_bnctx);
    unsigned char buf[64];
    vg_ocl_put_point(buf, poffset);
    vg_ocl_write_arg_buffer(vocp, 0, A_INCREMENT, buf);
    
    {
        // self-test
        EC_POINT * pnt = EC_POINT_new(pgroup);
        EC_POINT_mul(pgroup, pnt, EC_KEY_get0_private_key(pkey), NULL, NULL, vxcp->vxc_bnctx);
        EC_POINT * pnt1 = EC_POINT_new(pgroup);
        EC_POINT_add(pgroup, pnt1, pnt, poffset, vxcp->vxc_bnctx); // pnt1 = pnt + poffset
        BIGNUM * bn_round = BN_new();
        BN_set_word(bn_round, round);
        BIGNUM * bn_pkey_round = BN_new();
        BN_sub(bn_pkey_round, EC_KEY_get0_private_key(pkey), bn_round);
        EC_POINT * pnt2 = EC_POINT_new(pgroup);
        // pnt2 = (key-round)*gen
        EC_POINT_mul(pgroup, pnt2, bn_pkey_round, NULL, NULL, vxcp->vxc_bnctx);
        
        assert(!EC_POINT_cmp(pgroup, pnt1, pnt2, vxcp->vxc_bnctx));
        
        BIGNUM *x = BN_new(), *y = BN_new();
        EC_POINT_get_affine_coordinates_GFp(pgroup, poffset, x, y, vxcp->vxc_bnctx);
        BN_MONT_CTX * mont = BN_MONT_CTX_new();
        BIGNUM *mod = NULL;
        if (!BN_hex2bn(&mod, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f")) {
            printf("wrong hex\n");
            return NULL;
        }
        BN_MONT_CTX_set(mont, mod, vxcp->vxc_bnctx);
        BN_to_montgomery(x,x,mont,vxcp->vxc_bnctx);
        BN_to_montgomery(y,y,mont,vxcp->vxc_bnctx);
        //printf("CPU increment(X): %x %x %x %x %x %x %x %x\n", LIMBS(x));
        //printf("CPU increment(Y): %x %x %x %x %x %x %x %x\n", LIMBS(y));

    }
    
	vxcp->vxc_binres[0] = vcp->vc_addrtype;

	gettimeofday(&tvstart, NULL);
        
	vocp->voc_rekey_func(vocp);

	vg_exec_context_upgrade_lock(vxcp);

	for (int iteration=0; ; iteration++) {
        // key -= round
        BN_set_word(&vxcp->vxc_bntmp, round);
        BN_sub(&vxcp->vxc_bntmp2,
               EC_KEY_get0_private_key(pkey),
               &vxcp->vxc_bntmp);
        EC_KEY_set_private_key(pkey, &vxcp->vxc_bntmp2);
        //vg_set_privkey(&vxcp->vxc_bntmp2, pkey);
        
		if (vcp->vc_halt)
			halt = 1;
		if (halt)
			break;

		gettimeofday(&tv, NULL);
		
		if (!vg_ocl_kernel_start(vocp, 0, round, vocp->voc_ocl_invsize, &vxcp->vxc_bntmp2, iteration==0))
			halt = 1;
        
		// Wait for the GPU to complete its job
		if (vcp->vc_verbose > 1) printf("\nWait for the GPU to complete its job");
		if (!vg_ocl_kernel_wait(vocp, slot))
			halt = 1;
		
		/* Call the result check function */
		switch (vg_ocl_prefix_check(vocp, 0, &vxcp->vxc_bntmp2)) {
		case 1:
			break;
		case 2:
			halt = 1;
			break;
		default:
			break;
		}

		vg_output_timing(vcp, round, &tvstart);
		//vg_exec_context_yield(vxcp);

	    if (vcp->vc_verbose > 1) printf("\nsave the priv. key to a file");
        if (save_file_name && (iteration+1)%10 == 0) {
            //vg_encode_privkey(vxcp->vxc_key, vcp->vc_privtype, privkey_buf);
            FILE *sf = fopen(save_file_name, "w");
            if (sf) {
                char *privkey_buf = BN_bn2hex(EC_KEY_get0_private_key(vxcp->vxc_key));
                fputs(privkey_buf, sf);
                fputs("\n", sf);
                fclose(sf);
                OPENSSL_free(privkey_buf);
            }
        }
	}

	if (halt) {
		if (vcp->vc_verbose > 1) {
			printf("Halting...");
			fflush(stdout);
		}
	}

	vg_exec_context_yield(vxcp);

	if (poffset)
		EC_POINT_free(poffset);

	/* Release the argument buffers */
	vg_ocl_free_args(vocp);
	vocp->voc_halt = 0;
	vocp->voc_ocl_slot = -1;
	vg_context_thread_exit(vcp);
	return NULL;
}


/*
 * OpenCL platform/device selection junk
 */

int
get_device_list(cl_platform_id pid, cl_device_id **list_out)
{
	cl_uint nd;
	cl_int res;
	cl_device_id *ids;
	res = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
	if (res != CL_SUCCESS) {
		vg_ocl_error(NULL, res, "clGetDeviceIDs(0)");
		*list_out = NULL;
		return -1;
	}
	if (nd) {
		ids = (cl_device_id *) malloc(nd * sizeof(*ids));
		if (ids == NULL) {
			fprintf(stderr, "Could not allocate device ID list\n");
			*list_out = NULL;
			return -1;
		}
		res = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, nd, ids, NULL);
		if (res != CL_SUCCESS) {
			vg_ocl_error(NULL, res, "clGetDeviceIDs(n)");
			free(ids);
			*list_out = NULL;
			return -1;
		}
		*list_out = ids;
	}
	return nd;
}

void
show_devices(cl_platform_id pid, cl_device_id *ids, int nd, int base)
{
	int i;
	char nbuf[128];
	char vbuf[128];
	size_t len;
	cl_int res;

	for (i = 0; i < nd; i++) {
		res = clGetDeviceInfo(ids[i], CL_DEVICE_NAME,
				      sizeof(nbuf), nbuf, &len);
		if (res != CL_SUCCESS)
			continue;
		if (len >= sizeof(nbuf))
			len = sizeof(nbuf) - 1;
		nbuf[len] = '\0';
		res = clGetDeviceInfo(ids[i], CL_DEVICE_VENDOR,
				      sizeof(vbuf), vbuf, &len);
		if (res != CL_SUCCESS)
			continue;
		if (len >= sizeof(vbuf))
			len = sizeof(vbuf) - 1;
		vbuf[len] = '\0';
		fprintf(stderr, "  %d: [%s] %s\n", i + base, vbuf, nbuf);
	}
}

cl_device_id
get_device(cl_platform_id pid, int num)
{
	int nd;
	cl_device_id id, *ids;

	nd = get_device_list(pid, &ids);
	if (nd < 0) {
		printf("can't get_device_list\n");
		return NULL;
	}
	if (!nd) {
		fprintf(stderr, "No OpenCL devices found\n");
		return NULL;
	}
	if (num < 0) {
		num = 0;
	}
	if (num < nd) {
		id = ids[num];
		free(ids);
		return id;
	}
	free(ids);
	return NULL;
}

int
get_platform_list(cl_platform_id **list_out)
{
	cl_uint np;
	cl_int res;
	cl_platform_id *ids;
	res = clGetPlatformIDs(0, NULL, &np);
	if (res != CL_SUCCESS) {
		vg_ocl_error(NULL, res, "clGetPlatformIDs(0)");
		*list_out = NULL;
		return -1;
	}
	if (np) {
		ids = (cl_platform_id *) malloc(np * sizeof(*ids));
		if (ids == NULL) {
			fprintf(stderr,
				"Could not allocate platform ID list\n");
			*list_out = NULL;
			return -1;
		}
		res = clGetPlatformIDs(np, ids, NULL);
		if (res != CL_SUCCESS) {
			vg_ocl_error(NULL, res, "clGetPlatformIDs(n)");
			free(ids);
			*list_out = NULL;
			return -1;
		}
		*list_out = ids;
	}
	return np;
}

void
show_platforms(cl_platform_id *ids, int np, int base)
{
	int i;
	char nbuf[128];
	char vbuf[128];
	size_t len;
	cl_int res;

	for (i = 0; i < np; i++) {
		res = clGetPlatformInfo(ids[i], CL_PLATFORM_NAME,
					sizeof(nbuf), nbuf, &len);
		if (res != CL_SUCCESS) {
			vg_ocl_error(NULL, res, "clGetPlatformInfo(NAME)");
			continue;
		}
		if (len >= sizeof(nbuf))
			len = sizeof(nbuf) - 1;
		nbuf[len] = '\0';
		res = clGetPlatformInfo(ids[i], CL_PLATFORM_VENDOR,
					sizeof(vbuf), vbuf, &len);
		if (res != CL_SUCCESS) {
			vg_ocl_error(NULL, res, "clGetPlatformInfo(VENDOR)");
			continue;
		}
		if (len >= sizeof(vbuf))
			len = sizeof(vbuf) - 1;
		vbuf[len] = '\0';
		fprintf(stderr, "%d: [%s] %s\n", i + base, vbuf, nbuf);
	}
}

cl_platform_id
get_platform(int num)
{
	int np;
	cl_platform_id id, *ids;

	np = get_platform_list(&ids);
	if (np < 0)
		return NULL;
	if (!np) {
		fprintf(stderr, "No OpenCL platforms available\n");
		return NULL;
	}
	if (num < 0) {
		if (np == 1)
			num = 0;
		else
			num = np;
	}
	if (num < np) {
		id = ids[num];
		free(ids);
		return id;
	}
	free(ids);
	return NULL;
}

void
vg_ocl_enumerate_devices(void)
{
	cl_platform_id *pids;
	cl_device_id *dids;
	int np, nd, i;

	np = get_platform_list(&pids);
	if (!np) {
		fprintf(stderr, "No OpenCL platforms available\n");
		return;
	}
	fprintf(stderr, "Available OpenCL platforms:\n");
	for (i = 0; i < np; i++) {
		show_platforms(&pids[i], 1, i);
		nd = get_device_list(pids[i], &dids);
		if (!nd) {
			fprintf(stderr, "  -- No devices\n");
		} else {
			show_devices(pids[i], dids, nd, 0);
		}
	}
}

cl_device_id
get_opencl_device(int platformidx, int deviceidx)
{
	cl_platform_id pid;
	cl_device_id did = NULL;

	pid = get_platform(platformidx);
	if (pid) {
		did = get_device(pid, deviceidx);
		if (did)
			return did;
		else
			printf("cant get_device(%d)\n", deviceidx);
	}
	printf("cant get_platform(%d)\n", platformidx);
	return NULL;
}

#define min(a, b) ((a)<(b)? a: b)

vg_ocl_context_t *
vg_ocl_context_new(vg_context_t *vcp,
		   int platformidx, int deviceidx, int safe_mode, int verify,
		   int worksize, int nthreads, int nrows, int ncols,
		   int invsize, char * start)
{
	cl_device_id did;
	int round, full_threads, wsmult;
	cl_ulong memsize, allocsize;
	vg_ocl_context_t *vocp;

	/* Find the device */
	did = get_opencl_device(platformidx, deviceidx);
	if (!did) {
		printf("Can't get_opencl_device\n");
		return 0;
	}

	vocp = (vg_ocl_context_t *) malloc(sizeof(*vocp));
	if (!vocp)
		return NULL;

	/* Open the device and compile the kernel */
	if (!vg_ocl_init(vcp, vocp, did, safe_mode)) {
		printf("Can't vg_ocl_init\n");
		free(vocp);
		return NULL;
	}

	/*
	 * nrows: number of point rows per job
	 * ncols: number of point columns per job
	 * invsize: number of modular inversion tasks per job
	 *    (each task performs (nrows*ncols)/invsize inversions)
	 * nslots: number of kernels
	 *    (create two, keep one running while we service the other or wait)
	 */

	if (!nthreads) {
		/* Pick nthreads sufficient to saturate one compute unit */
		if (vg_ocl_device_gettype(vocp->voc_ocldid) &
		    CL_DEVICE_TYPE_CPU)
			nthreads = 1;
		else
			nthreads = vg_ocl_device_getsizet(vocp->voc_ocldid,
					CL_DEVICE_MAX_WORK_GROUP_SIZE);
	}

	full_threads = vg_ocl_device_getsizet(vocp->voc_ocldid,
					      CL_DEVICE_MAX_COMPUTE_UNITS);
	full_threads *= nthreads;

	/*
	 * The work size selection is complicated, and the most
	 * important factor is the batch size of the heap_invert kernel.
	 * Each value added to the batch trades one complete modular
	 * inversion for four multiply operations.  Ideally the work
	 * size would be as large as possible.  The practical limiting
	 * factors are:
	 * 1. Available memory
	 * 2. Responsiveness and operational latency
	 *
	 * We take a naive approach and limit batch size to a point of
	 * sufficiently diminishing returns, hoping that responsiveness
	 * will be sufficient.
	 *
	 * The measured value for the OpenSSL implementations on my CPU
	 * is 80:1.  This causes heap_invert to get batches of 20 or so
	 * for free, and receive 10% incremental returns at 200.  The CPU
	 * work size is therefore set to 256.
	 *
	 * The ratio on most GPUs with the oclvanitygen implementations
	 * is closer to 500:1, and larger batches are required for
	 * good performance.
	 */
	if (!worksize) {
		if (vg_ocl_device_gettype(vocp->voc_ocldid) &
		    CL_DEVICE_TYPE_GPU)
			worksize = 2048;
		else
			worksize = 256;
	}
	
	cl_ulong bitmapsize;
	memsize = vg_ocl_device_getulong(vocp->voc_ocldid, CL_DEVICE_GLOBAL_MEM_SIZE);
	allocsize = vg_ocl_device_getulong(vocp->voc_ocldid, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
	
	if (vcp->vc_verbose>1) printf("allocsize=%ld, memsize=%ld\n", allocsize, memsize);
	
	if (!ncols) {
		bitmapsize = min(allocsize, memsize/2);
		memsize -= bitmapsize;
		memsize /= 2;
		ncols = full_threads;
		nrows = 2;
		/* Find row and column counts close to sqrt(full_threads) */
		while ((ncols > nrows) && !(ncols & 1)) {
			ncols /= 2;
			nrows *= 2;
		}

		/*
		 * Increase row & column counts to satisfy work size
		 * multiplier or fill available memory.
		 */
		wsmult = 1;
		while ((!worksize || ((wsmult * 2) <= worksize)) &&
		       ((ncols * nrows * 2 * 128) < memsize) &&
		       ((ncols * nrows * 2 * 64) < allocsize)) {
			if (ncols > nrows)
				nrows *= 2;
			else
				ncols *= 2;
			wsmult *= 2;
		}
	} else {
		bitmapsize = min(allocsize, memsize-(ncols * nrows * 2 * 128));
	}

	round = nrows * ncols;
	
	if (!invsize) {
		invsize = 2;
		while (!(round % (invsize << 1)) &&
		       ((round / invsize) > full_threads))
			invsize <<= 1;
	}

	if (vcp->vc_verbose > 1) {
		fprintf(stderr, "Grid size: %dx%d\n", ncols, nrows);
		fprintf(stderr, "Modular inverse: %d threads, %d ops each\n",
			round/invsize, invsize);
		fprintf(stderr, "Bitmap size (bytes): %ld\n", bitmapsize);
	}

	if ((round % invsize) || !is_pow2(invsize) || (invsize < 2)) {
		if (vcp->vc_verbose <= 1) {
			fprintf(stderr, "Grid size: %dx%d\n", ncols, nrows);
			fprintf(stderr,
				"Modular inverse: %d threads, %d ops each\n",
				round/invsize, invsize);
		}
		if (round % invsize)
			fprintf(stderr,
				"Modular inverse work size must "
				"evenly divide points\n");
		else
			fprintf(stderr,
				"Modular inverse work per task (%d) "
				"must be a power of 2\n", invsize);
		goto out_fail;
	}

	vocp->voc_ocl_rows = nrows;
	vocp->voc_ocl_cols = ncols;
	vocp->voc_ocl_invsize = invsize;
	vocp->voc_ocl_bitmap_size = bitmapsize*(long)8; // bits
	//printf("vocp->voc_ocl_bitmap_size=%lld bitmapsize=%lld\n", vocp->voc_ocl_bitmap_size, bitmapsize);
	
	int addrtype = 128;
	if (!vg_decode_privkey_any(vocp->base.vxc_key, &addrtype, start, NULL)) {
		BIGNUM *pk = NULL;
		if (!BN_hex2bn(&pk, start)) {
			fprintf(stderr, "Wrong key format: %s\n", start);
			goto out_fail;
		}
		vg_set_privkey(pk, vocp->base.vxc_key);
		BN_free(pk);
	}
	
	return vocp;

out_fail:
	vg_ocl_context_free(vocp);
	return NULL;
}

vg_ocl_context_t *
vg_ocl_context_new_from_devstr(vg_context_t *vcp, const char *devstr,
			       int safemode, int verify, char * start)
{
	int platformidx, deviceidx;
	int worksize = 0, nthreads = 0, nrows = 0, ncols = 0, invsize = 0;

	char *dsd, *part, *part2, *param;

	dsd = strdup(devstr);
	if (!dsd)
		return NULL;

	part = strtok(dsd, ",");

	part2 = strchr(part, ':');
	if (!part2) {
		fprintf(stderr, "Invalid device specifier '%s'\n", part);
		free(dsd);
		return NULL;
	}

	*part2 = '\0';
	platformidx = atoi(part);
	deviceidx = atoi(part2 + 1);

	while ((part = strtok(NULL, ",")) != NULL) {
		param = strchr(part, '=');
		if (!param) {
			fprintf(stderr, "Unrecognized parameter '%s'\n", part);
			continue;
		}

		*param = '\0';
		param++;

		if (!strcmp(part, "grid")) {
			ncols = strtol(param, &part2, 0);
			if (part2 && *part2 == 'x') {
				nrows = strtol(part2+1, NULL, 0);
			}
			if (!nrows || !ncols) {
				fprintf(stderr,
					"Invalid grid size '%s'\n", param);
				nrows = 0;
				ncols = 0;
				continue;
			}
		}

		else if (!strcmp(part, "invsize")) {
			invsize = atoi(param);
			if (!invsize) {
				fprintf(stderr,
					"Invalid modular inverse size '%s'\n",
					param);
				continue;
			}
			if (invsize & (invsize - 1)) {
				fprintf(stderr,
					"Modular inverse size %d must be "
					"a power of 2\n", invsize);
				invsize = 0;
				continue;
			}
		}

		else if (!strcmp(part, "threads")) {
			nthreads = atoi(param);
			if (nthreads == 0) {
				fprintf(stderr,
					"Invalid thread count '%s'\n", param);
				continue;
			}
		}

		else {
			fprintf(stderr, "Unrecognized parameter '%s'\n", part);
		}
	}

	free(dsd);

	return vg_ocl_context_new(vcp, platformidx, deviceidx, safemode,
				  verify, worksize, nthreads, nrows, ncols,
				  invsize, start);
}


void
vg_ocl_context_free(vg_ocl_context_t *vocp)
{
	vg_ocl_del(vocp);
	free(vocp);
}
