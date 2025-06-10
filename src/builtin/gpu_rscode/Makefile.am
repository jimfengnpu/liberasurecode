lib_LTLIBRARIES = libgpu_rscode.la
# libgpu_rscode params
libgpu_rscode_la_SOURCES = matrix.cu encode.cu cpu_decode.c decode.cu gpu_rscode.c
libgpu_rscode_la_CPPFLAGS = -I$(top_srcdir)/include/gpu_rscode @GCOV_FLAGS@

NVCCFLAGS = @NVCCFLAGS@

# Version format  (C - A).(A).(R) for C:R:A input
libgpu_rscode_la_LDFLAGS = @GCOV_LDFLAGS@ -rpath '$(libdir)' -version-info 1:0:0

.cu.o:
	$(NVCC) -o $@ -c $< $(CCFLAGS) $(NVCCFLAGS)
.c.o:
	$(CC) -o $@ -c $< $(CCFLAGS)

MOSTLYCLEANFILES = *.gcda *.gcno *.gcov
