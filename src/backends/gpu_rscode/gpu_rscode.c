#include "erasurecode.h"
#include "erasurecode_backend.h"

#define LIBGPU_RSCODE_LIB_MAJOR 1
#define LIBGPU_RSCODE_LIB_MINOR 0
#define LIBGPU_RSCODE_LIB_REV 0
#define LIBGPU_RSCODE_LIB_VER_STR "1.0"
#define LIBGPU_RSCODE_LIB_NAME "libgpu_rscode"
#if defined(__MACOS__) || defined(__MACOSX__) || defined(__OSX__) || defined(__APPLE__)
#define LIBGPU_RSCODE_SO_NAME "libgpu_rscode" LIBERASURECODE_SO_SUFFIX ".dylib"
#else
#define LIBGPU_RSCODE_SO_NAME "libgpu_rscode" LIBERASURECODE_SO_SUFFIX ".so.1"
#endif

/* declarations */
struct ec_backend_common backend_libgpu_rscode;

typedef int (*libgpurscode_encode_func)(uint8_t *, char **, char **, int, int, int);
typedef int (*libgpurscode_decode_func)(uint8_t *, char **, char **, int, int, int *, int);
typedef int (*libgpurscode_reconstruct_func)(uint8_t *, char **, char **, int, int, int *, int, int);
typedef void (*libgpurscode_init_func)(int, int, uint8_t**);


struct libgpu_rscode_descriptor {
    libgpurscode_init_func init;
    libgpurscode_encode_func encode;
    libgpurscode_decode_func decode;
    libgpurscode_reconstruct_func reconstruct;

    uint8_t *generator_matrix;
    int k;
    int m;
};

static void * libgpu_rscode_init(struct ec_backend_args *args, void *backend_sohandle)
{
    struct libgpu_rscode_descriptor *desc = NULL;
    desc = (struct libgpu_rscode_descriptor *)
           malloc(sizeof(struct libgpu_rscode_descriptor));
    if (NULL == desc) {
        return NULL;
    }

    desc->k = args->uargs.k;
    desc->m = args->uargs.m;

     /*
     * ISO C forbids casting a void* to a function pointer.
     * Since dlsym return returns a void*, we use this union to
     * "transform" the void* to a function pointer.
     */
    union {
        libgpurscode_init_func initp;
        libgpurscode_encode_func encodep;
        libgpurscode_decode_func decodep;
        libgpurscode_reconstruct_func reconstructp;
        void *vptr;
    } func_handle = {.vptr = NULL};


    /* fill in function addresses */
    func_handle.vptr = NULL;
    func_handle.vptr = dlsym(backend_sohandle, "libgpu_rscode_init");
    desc->init = func_handle.initp;
    if (NULL == desc->init) {
        goto error;
    }

    func_handle.vptr = NULL;
    func_handle.vptr = dlsym(backend_sohandle, "libgpu_rscode_encode");
    desc->encode = func_handle.encodep;
    if (NULL == desc->encode) {
        goto error;
    }

    func_handle.vptr = NULL;
    func_handle.vptr = dlsym(backend_sohandle, "libgpu_rscode_decode");
    desc->decode = func_handle.decodep;
    if (NULL == desc->decode) {
        goto error;
    }

    func_handle.vptr = NULL;
    func_handle.vptr = dlsym(backend_sohandle, "libgpu_rscode_reconstruct");
    desc->reconstruct = func_handle.reconstructp;
    if (NULL == desc->reconstruct) {
        goto error;
    }

    desc->init(desc->k, desc->m, &desc->generator_matrix);

    if (NULL == desc->generator_matrix) {
        goto error;
    }

    return desc;

error:
    free(desc);

    return NULL;
}

static int libgpu_rscode_exit(void *desc)
{
    struct libgpu_rscode_descriptor *gpurscode_desc =
        (struct libgpu_rscode_descriptor*)desc;
    free(gpurscode_desc->generator_matrix);
    free(gpurscode_desc);
    return 0;
}

static int libgpu_rscode_encode(void *desc, char **data, char **parity, 
        int blocksize)
{
    struct libgpu_rscode_descriptor *gpurscode_desc =
        (struct libgpu_rscode_descriptor*)desc;
    gpurscode_desc->encode(gpurscode_desc->generator_matrix, data, parity, 
        gpurscode_desc->k, gpurscode_desc->m, blocksize);
    return 0;
}

static int libgpu_rscode_decode(void *desc, char **data, char **parity, 
        int *missing_idxs, int blocksize)
{
    struct libgpu_rscode_descriptor *gpurscode_desc =
        (struct libgpu_rscode_descriptor*)desc;
    gpurscode_desc->decode(gpurscode_desc->generator_matrix, data, parity, 
        gpurscode_desc->k, gpurscode_desc->m, missing_idxs, blocksize);
}

static int libgpu_rscode_reconstruct(void *desc, char **data, char **parity, 
        int *missing_idxs, int destination_idx, int blocksize)
{
    struct libgpu_rscode_descriptor *gpurscode_desc =
        (struct libgpu_rscode_descriptor*)desc;
    gpurscode_desc->reconstruct(gpurscode_desc->generator_matrix, data, parity, 
        gpurscode_desc->k, gpurscode_desc->m, missing_idxs, destination_idx, blocksize);
}

static int libgpu_rscode_min_fragments(void *desc, int *missing_idxs,
        int *fragments_to_exclude, int *fragments_needed)
{
    struct libgpu_rscode_descriptor *gpurscode_desc =
        (struct libgpu_rscode_descriptor*)desc;

    uint64_t exclude_bm = convert_list_to_bitmap(fragments_to_exclude);
    uint64_t missing_bm = convert_list_to_bitmap(missing_idxs) | exclude_bm;
    int i;
    int j = 0;
    int ret = -1;

    for (i = 0; i < (gpurscode_desc->k + gpurscode_desc->m); i++) {
        if (!(missing_bm & (1 << i))) {
            fragments_needed[j] = i;
            j++;
        }
        if (j == gpurscode_desc->k) {
            ret = 0;
            fragments_needed[j] = -1;
            break;
        }
    }

    return ret;
}

static int libgpu_rscode_element_size(void *desc) {
    return 8;
}

static bool libgpu_rscode_is_compatible_with(uint32_t version) {
    return version == backend_liberasurecode_rs_vand.ec_backend_version;
}

static struct ec_backend_op_stubs libgpu_rscode_op_stubs = {
    .INIT                       = libgpu_rscode_init,
    .EXIT                       = libgpu_rscode_exit,
    .ENCODE                     = libgpu_rscode_encode,
    .DECODE                     = libgpu_rscode_decode,
    .FRAGSNEEDED                = libgpu_rscode_min_fragments,
    .RECONSTRUCT                = libgpu_rscode_reconstruct,
    .ELEMENTSIZE                = libgpu_rscode_element_size,
    .ISCOMPATIBLEWITH           = libgpu_rscode_is_compatible_with,
    .GETMETADATASIZE            = get_backend_metadata_size_zero,
    .GETENCODEOFFSET            = get_encode_offset_zero,
};

__attribute__ ((visibility ("internal")))
struct ec_backend_common backend_liberasurecode_rs_vand = {
    .id                         = EC_BACKEND_LIBERASURECODE_RS_VAND,
    .name                       = LIBGPU_RSCODE_LIB_NAME,
    .soname                     = LIBGPU_RSCODE_SO_NAME,
    .soversion                  = LIBGPU_RSCODE_LIB_VER_STR,
    .ops                        = &libgpu_rscode_op_stubs,
    .ec_backend_version         = _VERSION(LIBGPU_RSCODE_LIB_MAJOR,
                                           LIBGPU_RSCODE_LIB_MINOR,
                                           LIBGPU_RSCODE_LIB_REV),
};