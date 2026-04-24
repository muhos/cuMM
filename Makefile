CUDA_PATH ?= /usr/local/cuda

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CUDA_PATH),NONE)
$(error cannot find CUDA local directory)
endif
endif

# device/host compilers (nvcc is the master)

CXX  := g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

# compiler flags

RELOC     := -dc
CCFLAGS   := -std=c++20 -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20 -lineinfo -Xptxas -v #-maxrregcount=62 #-ptx 

# Common includes 
INCLUDES  := -I../../common/inc
EXTRALIB  :=

BINARY := cuMM

# debug build flags
ifeq ($(debug),1)
      NVCCFLAGS += -g -G -DDEBUG
else  ifeq ($(assert),1)
      NVCCFLAGS += -O3
else
      NVCCFLAGS += -O3 -DNDEBUG -lcublas
endif

# combine all flags
ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# gencode arguments

GENCODE_FLAGS := -arch=native

all: $(BINARY)

$(BINARY): src/$(BINARY).cu
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $<

clean:
	rm -f $(BINARY)

.PHONY: all clean
