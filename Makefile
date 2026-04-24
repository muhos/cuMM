CUDA_PATH ?= /usr/local/cuda

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CUDA_PATH),NONE)
$(error cannot find CUDA local directory)
endif
endif

CXX  := g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CCFLAGS   := -std=c++20 -fdiagnostics-show-option
NVCCFLAGS := -m64 -std=c++20 -lineinfo

INCLUDES  := -I../../common/inc
EXTRALIB  :=

BINARY := cuMM

ifeq ($(debug),1)
      NVCCFLAGS += -g -G -DDEBUG
else ifeq ($(assert),1)
      NVCCFLAGS += -O3
else
      NVCCFLAGS += -O3 -DNDEBUG
      EXTRALIB  += -lcublas
endif

ifdef reg
      NVCCFLAGS += -maxrregcount=$(reg)
endif

ifeq ($(count),1)
      NVCCFLAGS += -Xptxas -v --resource-usage
endif

ALL_CCFLAGS := $(NVCCFLAGS) $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_LDFLAGS := $(ALL_CCFLAGS) $(addprefix -Xlinker ,$(LDFLAGS)) $(EXTRALIB)

GENCODE_FLAGS := -arch=native

MAINCU  := src/$(BINARY).cu
ALLCU   := $(sort $(wildcard src/*.cu src/*/*.cu))
OTHERCU := $(filter-out $(MAINCU),$(ALLCU))

PTX_DIR   := ptx
PTX_FILES := $(patsubst src/%.cu,$(PTX_DIR)/%.ptx,$(OTHERCU) $(MAINCU))

ifeq ($(ptx),1)
all: $(PTX_FILES)
else
all: $(BINARY)
endif

$(BINARY): $(OTHERCU) $(MAINCU)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $^

$(PTX_DIR)/%.ptx: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -ptx -o $@ $<

clean:
	rm -f $(BINARY) src/*.o 
	rm -rf $(PTX_DIR)

.PHONY: all clean
