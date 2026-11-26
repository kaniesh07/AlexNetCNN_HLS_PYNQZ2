# run_hls.tcl -- Vitis HLS 2024.2 flow for the AlexNet-C10 (CIFAR-10) accelerator
#
# From the hls/ directory:
#   vitis-run --mode hls --tcl scripts/run_hls.tcl          (2024.2 unified CLI)
#   vitis_hls -f scripts/run_hls.tcl                        (legacy command, if present)
# Environment:
#   HLS_STEPS  default "csim csynth export"  (add cosim:  "csim csynth cosim export")
#   CLK_MHZ    default 150  (use 100 if csynth/implementation misses timing)
#   CIFAR_N    images for csim (default 20); cosim always uses 2
# Prerequisite: python model/export_hls.py   (creates include/cifar_weights.h + testbench/data)
#
# Outputs
#   cifar_hls/solution1/syn/report/cifar_accel_csynth.rpt     latency / resources / II
#   cifar_hls/solution1/impl/ip/                               IP repository for Vivado
#   cifar_hls/solution1/impl/misc/drivers/cifar_accel_v1_0/src/xcifar_accel_hw.h
#                                                              AUTHORITATIVE register map

set steps "csim csynth export"
if {[info exists ::env(HLS_STEPS)]} { set steps $::env(HLS_STEPS) }
set clk_mhz 150
if {[info exists ::env(CLK_MHZ)]} { set clk_mhz $::env(CLK_MHZ) }
set here [pwd]
if {![file exists [file join $here include cifar_weights.h]]} {
    error "include/cifar_weights.h missing -- run: python model/export_hls.py"
}

open_project -reset cifar_hls
set_top cifar_accel
add_files src/cifar_accel.cpp -cflags "-I$here/include -std=c++14"
add_files -tb testbench/tb_cifar_accel.cpp \
    -cflags "-I$here/include -std=c++14 -DTB_DATA_DIR=\\\"$here/testbench/data\\\""

open_solution -reset solution1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period [expr {1000.0 / $clk_mhz}] -name default
set_clock_uncertainty 12.5%
config_rtl -reset control                        ;# reset only control logic: less fan-out, less power
config_compile -pipeline_style flp               ;# flushable pipelines
config_export -format ip_catalog -rtl verilog -vendor xilinx.com -library hls \
              -version 1.0 -display_name "AlexNet-C10 CIFAR-10 accelerator"

if {[lsearch $steps csim] >= 0} {
    puts "=== C simulation ==="
    csim_design
}
if {[lsearch $steps csynth] >= 0} {
    puts "=== C synthesis @ $clk_mhz MHz ==="
    csynth_design
}
if {[lsearch $steps cosim] >= 0} {
    puts "=== C/RTL co-simulation (2 images) ==="
    set ::env(CIFAR_N) 2
    cosim_design -rtl verilog -trace_level none
}
if {[lsearch $steps export] >= 0} {
    puts "=== export IP ==="
    export_design -format ip_catalog -rtl verilog
}
exit
