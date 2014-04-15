flags=-DPRETTY_PRINT -G -DCPU_KD -DGPU_KD -DGPU -lm -arch sm_13
kdkmeans : kdkmeans.cu kd.cu kd.h common.h
	nvcc -o kdkmeans kdkmeans.cu $(flags)

# Can remove -G flag for GPU. But GPU_KD code is not working without
# -G. So -G needs to be there for GPU_KD. (-G specifies -O0).

clean : 
	-rm -rf *.o *~ kdkmeans
