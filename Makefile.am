# Top-level liberasurecode automake configuration
SUBDIRS = src test doc

EXTRA_DIST = autogen.sh get_flags_from_cpuid.c cudalt.py

INCLUDE = -I$(abs_top_builddir)/include \
		  -I$(abs_top_builddir)/include/erasurecode \
		  -I$(abs_top_builddir)/include/xor_codes

AM_CPPFLAGS = $(CPPFLAGS) $(INCLUDE)
AM_CPPFLAGS += -Werror -Wall

AM_CFLAGS = -fPIC $(AM_CPPFLAGS) @GCOV_FLAGS@ -L/usr/local/lib

thisincludedir = $(includedir)/liberasurecode
thisinclude_HEADERS = \
	include/erasurecode/alg_sig.h \
	include/erasurecode/erasurecode.h \
	include/erasurecode/erasurecode_backend.h \
	include/erasurecode/erasurecode_helpers.h \
	include/erasurecode/erasurecode_helpers_ext.h \
	include/erasurecode/erasurecode_log.h \
	include/erasurecode/erasurecode_preprocessing.h \
	include/erasurecode/erasurecode_postprocessing.h \
	include/erasurecode/erasurecode_stdinc.h \
	include/erasurecode/erasurecode_version.h \
	include/erasurecode/list.h \
	include/xor_codes/xor_hd_code_defs.h \
	include/xor_codes/xor_code.h \
	include/config_liberasurecode.h \
	include/rs_vand/rs_galois.h \
	include/rs_vand/liberasurecode_rs_vand.h

pkgconfig_DATA = erasurecode-$(LIBERASURECODE_API_VERSION).pc

install-exec-hook:
	-(mkdir -p $(DESTDIR)$(includedir) && \
	   cd $(DESTDIR)$(includedir) && \
	   rm -f erasurecode.h erasurecode_version.h \
	         erasurecode_stdinc.h erasurecode_helpers.h \
	         config_liberasurecode.h && \
	   $(LN_S) liberasurecode/erasurecode.h && \
	   $(LN_S) liberasurecode/erasurecode_version.h && \
	   $(LN_S) liberasurecode/erasurecode_stdinc.h && \
	   $(LN_S) liberasurecode/erasurecode_helpers.h && \
	   $(LN_S) liberasurecode/config_liberasurecode.h)

test: check
	@./test/liberasurecode_test
	@./test/alg_sig_test
	@./test/test_xor_hd_code
	@./test/libec_slap
 
LIBTOOL_COMMAND = $(LIBTOOL) --mode execute
VALGRIND_EXEC_COMMAND = $(LIBTOOL_COMMAND) valgrind --tool=memcheck \
	--error-exitcode=1 --leak-check=yes --track-fds=yes \
	--malloc-fill=A5 --free-fill=DE --fullpath-after=. --trace-children=yes

valgrind-test: check
	@$(VALGRIND_EXEC_COMMAND) ./test/alg_sig_test
	@$(VALGRIND_EXEC_COMMAND) ./test/liberasurecode_test
	@$(VALGRIND_EXEC_COMMAND) ./test/test_xor_hd_code
	@$(VALGRIND_EXEC_COMMAND) ./test/libec_slap

CLEANFILES = cscope.in.out cscope.out cscope.po.out

.PHONY: cscope
cscope:
	find src include -name "*.[ch]" > cscope.files
	cscope -q -b

MOSTLYCLEANFILES = *.gcda *.gcno *.gcov 
