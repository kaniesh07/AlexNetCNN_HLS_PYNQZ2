# create_project.tcl -- Vivado 2024.2 project: PYNQ-Z2 + Vitis HLS AlexNet-C10 IP
#
# Prerequisites
#   1. PYNQ-Z2 board files (docs/compatibility.md), or PYNQ_Z2_BOARD_REPO=<folder>
#   2. HLS IP exported: cd hls && vitis-run --mode hls --tcl scripts/run_hls.tcl
# Run from the repository root:
#   vivado -mode batch -source vivado/create_project.tcl
# Env: CLK_MHZ (default 150) -- must equal the clock used in run_hls.tcl

set root    [file normalize [file join [file dirname [info script]] ..]]
set proj    [file join $root build vivado]
set hls_ip  [file join $root hls cifar_hls solution1 impl ip]
set ::clk_mhz 150
if {[info exists ::env(CLK_MHZ)]} { set ::clk_mhz $::env(CLK_MHZ) }
if {[info exists ::env(PYNQ_Z2_BOARD_REPO)]} {
    set_param board.repoPaths [list $::env(PYNQ_Z2_BOARD_REPO)]
}

create_project cifar_hls_pynq $proj -part xc7z020clg400-1 -force
set bp [lindex [get_board_parts -quiet -latest_file_version {*pynq-z2*}] 0]
if {$bp eq ""} {
    error "PYNQ-Z2 board files not found (needed for the PS7 DDR/MIO preset); see docs/compatibility.md"
}
set_property board_part $bp [current_project]
set_property target_language Verilog [current_project]

if {![file isdirectory $hls_ip]} {
    error "HLS IP not found at $hls_ip -- run hls/scripts/run_hls.tcl first"
}
set_property ip_repo_paths [list $hls_ip] [current_project]
update_ip_catalog -rebuild
if {[llength [get_ipdefs -quiet xilinx.com:hls:cifar_accel:*]] == 0} {
    error "cifar_accel IP not visible in the IP catalog"
}
source [file join $root vivado create_block_design.tcl]
