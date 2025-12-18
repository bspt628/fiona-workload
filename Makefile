XLEN ?= 64

default: all

src_dir = ./
target_dir = ./build

#--------------------------------------------------------------------
# Build Mode: pk (default) or baremetal
#--------------------------------------------------------------------
# BAREMETAL=1 : Verilator/FPGA向けbare-metalビルド（htif_nano.specs使用）
# BAREMETAL=0 : Spike向けpkビルド（デフォルト）
#
# bare-metalビルドには-mcmodel=medanyでコンパイルされたツールチェーンを使用
BAREMETAL ?= 0

# Medany toolchain for bare-metal builds (built with --with-cmodel=medany)
RISCV_MEDANY_DIR ?= $(abspath ../toolchains/riscv-medany)

#--------------------------------------------------------------------
# Chipyard libgloss paths (for bare-metal builds)
#--------------------------------------------------------------------
CHIPYARD_DIR ?= $(abspath ../chipyard)
LIBGLOSS_UTIL = $(CHIPYARD_DIR)/toolchains/libgloss/util
LIBGLOSS_BUILD = $(CHIPYARD_DIR)/toolchains/libgloss/build

#--------------------------------------------------------------------
# Sources
#--------------------------------------------------------------------

applications = \
	hello_fiona		\
	math_palu		\
	math_ealu		\
	math_nlu		\
	math_misc		\
	nn_linear		\
	nn_conv			\
	nn_pad			\
	nn_pool			\
	mlp_iris		\
	mlp_iris_train	\
	mlp_iris_infer	\
	mlp_large		\
	mlp_mnist		\
	mlp_mnist_fp32	\
	elm_iris		\
	test_relu		\
	regression_benchmark	\
	mlp_mnist_photonic	\
	mlp_mnist_photonic_small	\
	test_transformer	\
	benchmark_transformer	\
	text_gen_transformer

#--------------------------------------------------------------------
# Build rules
#--------------------------------------------------------------------

incs  += -I$(src_dir)/lib $(addprefix -I$(src_dir)/app/, $(applications))
objs  :=

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf-
RISCV_GXX ?= $(RISCV_PREFIX)g++
RISCV_OBJDUMP ?= $(RISCV_PREFIX)objdump --disassemble-all --disassemble-zeroes --section=.text --section=.text.startup --section=.text.init --section=.data
RISCV_OBJCOPY ?= $(RISCV_PREFIX)objcopy -S --set-section-flags .bss=alloc,contents -O binary

#--------------------------------------------------------------------
# Build mode specific options
#--------------------------------------------------------------------

ifeq ($(BAREMETAL),1)
# Bare-metal build for Verilator/FPGA
# Uses medany toolchain and htif_nano.specs for HTIF support
target_dir = ./build-baremetal
RISCV_PREFIX := $(RISCV_MEDANY_DIR)/bin/riscv$(XLEN)-unknown-elf-
RISCV_GXX := $(RISCV_PREFIX)g++
RISCV_OBJDUMP := $(RISCV_PREFIX)objdump --disassemble-all --disassemble-zeroes --section=.text --section=.text.startup --section=.text.init --section=.data
RISCV_OBJCOPY := $(RISCV_PREFIX)objcopy -S --set-section-flags .bss=alloc,contents -O binary
RISCV_GXX_OPTS ?= -static -std=gnu++11 -O2 -ffast-math -fno-tree-loop-distribute-patterns \
	-specs=$(LIBGLOSS_UTIL)/htif_nano.specs \
	-L$(LIBGLOSS_UTIL) \
	-L$(LIBGLOSS_BUILD)
RISCV_LINK_OPTS ?= -static -lm -lstdc++
BUILD_MODE = baremetal
else
# pk (proxy kernel) build for Spike (default)
target_dir = ./build
RISCV_GXX_OPTS ?= -static -std=gnu++11 -O2 -ffast-math -fno-tree-loop-distribute-patterns
RISCV_LINK_OPTS ?= -static -lm -lstdc++
BUILD_MODE = pk
endif

RISCV_LINK ?= $(RISCV_GXX) $(incs)
RISCV_SIM ?= spike pk --isa=rv$(XLEN)gc --extension=fiona


define compile_template
$(target_dir)/$(1).elf: $(wildcard $(src_dir)/app/$(1)/*)
	@echo "Building $(1) [mode: $(BUILD_MODE)]"
	mkdir -p $(target_dir)
	$$(RISCV_GXX) $$(incs) $$(RISCV_GXX_OPTS) -o $$@ \
		$(wildcard $(src_dir)/app/$(1)/*.c) \
		$(wildcard $(src_dir)/app/$(1)/*.cc) \
		$(wildcard $(src_dir)/app/$(1)/*.S) \
		$(wildcard $(src_dir)/lib/math/*.c) \
		$(wildcard $(src_dir)/lib/math/*.cc) \
		$(wildcard $(src_dir)/lib/math/*.S) \
		$(wildcard $(src_dir)/lib/nn/*.c) \
		$(wildcard $(src_dir)/lib/nn/*.cc) \
		$(wildcard $(src_dir)/lib/nn/*.S) \
		$(wildcard $(src_dir)/lib/utils/*.c) \
		$(wildcard $(src_dir)/lib/utils/*.cc) \
		$(wildcard $(src_dir)/lib/utils/*.S) \
		$$(RISCV_LINK_OPTS)
	$$(RISCV_OBJDUMP) $$@ > $$(target_dir)/$(1).S
	$$(RISCV_OBJCOPY) $$@   $$(target_dir)/$(1).bin
endef

$(foreach app,$(applications),$(eval $(call compile_template,$(app))))

#------------------------------------------------------------
# Build and run benchmarks on riscv simulator


apps_bin  = $(addprefix $(target_dir)/, $(addsuffix .elf,  $(applications)))


apps: $(apps_bin)

artifacts += $(apps_bin)

#------------------------------------------------------------
# Default

all: apps

#------------------------------------------------------------
# Convenience targets

.PHONY: pk baremetal clean clean-all help

# Build all apps for pk (Spike)
pk:
	$(MAKE) BAREMETAL=0 all

# Build all apps for bare-metal (Verilator/FPGA)
baremetal:
	$(MAKE) BAREMETAL=1 all

# Clean current build mode
clean:
	rm -rf $(target_dir)

# Clean all build directories
clean-all:
	rm -rf ./build ./build-baremetal

# Help
help:
	@echo "FIONA Workload Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all apps for pk (Spike) [default]"
	@echo "  make pk           - Build all apps for pk (Spike)"
	@echo "  make baremetal    - Build all apps for bare-metal (Verilator/FPGA)"
	@echo "  make BAREMETAL=1  - Build all apps for bare-metal"
	@echo ""
	@echo "  make clean        - Clean current build directory"
	@echo "  make clean-all    - Clean all build directories"
	@echo ""
	@echo "Build Directories:"
	@echo "  ./build/          - pk builds (for Spike + proxy kernel)"
	@echo "  ./build-baremetal/ - bare-metal builds (for Verilator/FPGA)"
	@echo ""
	@echo "Examples:"
	@echo "  # Build hello_fiona for pk"
	@echo "  make build/hello_fiona.elf"
	@echo ""
	@echo "  # Build hello_fiona for bare-metal"
	@echo "  make BAREMETAL=1 build-baremetal/hello_fiona.elf"
	@echo ""
	@echo "  # Build all for both modes"
	@echo "  make pk && make baremetal"
