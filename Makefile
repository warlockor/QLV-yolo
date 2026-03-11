CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS ?=

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

BINDIR := build

ARMNN_INCLUDE_DIR ?= /usr/include
ARMNN_TFLITE_INCLUDE_DIR ?= /usr/include
ARMNN_LIB_DIR ?= /usr/lib

TFLITE_INCLUDE_DIR ?= /usr/include
TFLITE_LIB_DIR ?= /usr/lib
TFLITE_LIB_NAME ?= tensorflow-lite
TFLITE_GPU_LIB_NAME ?= tensorflowlite_gpu_delegate

COMMON_CXXFLAGS := $(CXXFLAGS) $(OPENCV_CFLAGS)
COMMON_LIBS := $(OPENCV_LIBS) -lpthread -ldl

.PHONY: all armnn tflite_cpu tflite_gpu clean help

all: armnn tflite_cpu tflite_gpu

$(BINDIR):
	mkdir -p $(BINDIR)

armnn: $(BINDIR)/yolo_armnn_opencl

$(BINDIR)/yolo_armnn_opencl: src/main.cpp | $(BINDIR)
	$(CXX) $(COMMON_CXXFLAGS) \
	-I$(ARMNN_INCLUDE_DIR) \
	-I$(ARMNN_TFLITE_INCLUDE_DIR) \
	$< -o $@ \
	-L$(ARMNN_LIB_DIR) -larmnn -larmnnTfLiteParser \
	$(COMMON_LIBS) $(LDFLAGS)

tflite_cpu: $(BINDIR)/yolo_tflite_cpu

$(BINDIR)/yolo_tflite_cpu: src/tflite_cpu_main.cpp | $(BINDIR)
	$(CXX) $(COMMON_CXXFLAGS) \
	-I$(TFLITE_INCLUDE_DIR) \
	$< -o $@ \
	-L$(TFLITE_LIB_DIR) -l$(TFLITE_LIB_NAME) \
	$(COMMON_LIBS) $(LDFLAGS)

tflite_gpu: $(BINDIR)/yolo_tflite_gpu_delegate

$(BINDIR)/yolo_tflite_gpu_delegate: src/tflite_gpu_main.cpp | $(BINDIR)
	$(CXX) $(COMMON_CXXFLAGS) \
	-I$(TFLITE_INCLUDE_DIR) \
	$< -o $@ \
	-L$(TFLITE_LIB_DIR) -l$(TFLITE_LIB_NAME) -l$(TFLITE_GPU_LIB_NAME) \
	$(COMMON_LIBS) $(LDFLAGS)

clean:
	rm -rf $(BINDIR)

help:
	@echo "Targets:"
	@echo "  make all          # build all backends"
	@echo "  make armnn        # build Arm NN + OpenCL binary"
	@echo "  make tflite_cpu   # build TensorFlow Lite CPU binary"
	@echo "  make tflite_gpu   # build TensorFlow Lite GPU delegate binary"
	@echo "  make clean        # remove build artifacts"
	@echo ""
	@echo "Override paths if libraries are not in system default locations:"
	@echo "  ARMNN_INCLUDE_DIR ARMNN_TFLITE_INCLUDE_DIR ARMNN_LIB_DIR"
	@echo "  TFLITE_INCLUDE_DIR TFLITE_LIB_DIR TFLITE_LIB_NAME TFLITE_GPU_LIB_NAME"
