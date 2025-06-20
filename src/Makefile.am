SUBDIRS = builtin/xor_codes builtin/null_code builtin/rs_vand builtin/gpu_rscode

lib_LTLIBRARIES = liberasurecode.la

INCLUDE = \
		-I$(abs_top_srcdir)/include/erasurecode \
		-I$(abs_top_srcdir)/include/xor_codes \
		-I$(abs_top_srcdir)/include/rs_vand \
		-I$(abs_top_srcdir)/include/isa_l \
		-I$(abs_top_srcdir)/include/shss \
		-I$(abs_top_srcdir)/include/gpu_rscode

AM_CPPFLAGS = $(CPPFLAGS) $(INCLUDE)
AM_CFLAGS = $(AM_CPPFLAGS)

# liberasurecode params
liberasurecode_la_SOURCES = \
		erasurecode.c \
		erasurecode_helpers.c \
		erasurecode_preprocessing.c \
		erasurecode_postprocessing.c \
		utils/chksum/crc32.c \
		utils/chksum/alg_sig.c \
		backends/null/null.c \
		backends/xor/flat_xor_hd.c \
		backends/jerasure/jerasure_rs_vand.c \
		backends/jerasure/jerasure_rs_cauchy.c \
		backends/isa-l/isa_l_common.c \
		backends/isa-l/isa_l_rs_vand.c \
		backends/isa-l/isa_l_rs_cauchy.c \
		backends/rs_vand/liberasurecode_rs_vand.c \
		backends/shss/shss.c \
		backends/phazrio/libphazr.c \
		backends/gpu_rscode/gpu_rscode.c

liberasurecode_la_CPPFLAGS = -Werror @GCOV_FLAGS@
liberasurecode_la_LIBADD = \
		builtin/null_code/libnullcode.la \
		builtin/xor_codes/libXorcode.la \
		builtin/rs_vand/liberasurecode_rs_vand.la \
		builtin/gpu_rscode/libgpu_rscode.la	\
		-lpthread -lm -lz @GCOV_LDFLAGS@

# Version format  (C - A).(A).(R) for C:R:A input
liberasurecode_la_LDFLAGS = -rpath '$(libdir)' -version-info @LIBERASURECODE_VERSION_INFO@

MOSTLYCLEANFILES = *.gcda *.gcno *.gcov utils/chksum/*.gcda utils/chksum/*.gcno utils/chksum/*.gcov \
                   backends/null/*.gcda backends/null/*.gcno backends/null/*.gcov  \
                   backends/xor/*.gcda backends/xor/*.gcno backends/xor/*.gcov  \
                   backends/jerasure/*.gcda backends/jerasure/*.gcno backends/jerasure/*.gcov \
                   backends/shss/*.gcda backends/shss/*.gcno backends/shss/*.gcov \
                   backends/rs_vand/*.gcda backends/rs_vand/*.gcno backends/rs_vand/*.gcov \
                   backends/phazrio/*.gcda backends/phazrio/*.gcno backends/phazrio/*.gcov
