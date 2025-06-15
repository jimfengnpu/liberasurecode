lib_LTLIBRARIES = libgpu_rscode.la
# libgpu_rscode params
libgpu_rscode_la_SOURCES = gpu_matrix.cu gpu_encode.cu cpu_decode.c gpu_decode.cu gpu_rscode.c
libgpu_rscode_la_CPPFLAGS = -I$(top_srcdir)/include/gpu_rscode @GCOV_FLAGS@ @CUDA_CFLAGS@
libgpu_rscode_la_LIBADD = -lcudart -lpthread -lstdc++
NVCCFLAGS = @NVCCFLAGS@

# Version format  (C - A).(A).(R) for C:R:A input
libgpu_rscode_la_LDFLAGS = @GCOV_LDFLAGS@ @CUDA_LIBS@ -rpath '$(libdir)' -version-info 1:0:0

.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS) $(libgpu_rscode_la_CPPFLAGS)
.cu.lo:
	$(top_srcdir)/cudalt.py $(top_builddir)/libtool $@ $(NVCC) $(NVCCFLAGS) --compiler-options=\"$(DEFAULT_INCLUDES) $(libgpu_rscode_la_CPPFLAGS) $(AM_CPPFLAGS) $(CPPFLAGS)\" -c $<  

MOSTLYCLEANFILES = *.gcda *.gcno *.gcov
