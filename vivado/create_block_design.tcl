# create_block_design.tcl -- block design "design_1" (sourced by create_project.tcl)
#
#  PS7 M_AXI_GP0 --smc_ctrl--> cifar_accel_0/s_axi_control @ 0x4000_0000 (64K)
#                         \--> axi_dma_0/S_AXI_LITE       @ 0x4040_0000 (64K)
#                          \-> axi_intc_0/s_axi           @ 0x4180_0000 (64K)
#  axi_dma_0/M_AXI_MM2S --smc_hp0--> PS7 S_AXI_HP0   (reads the 4 KB image only)
#  axi_dma_0/M_AXIS_MM2S --> cifar_accel_0/img        (AXI4-Stream, 32 bit)
#  {cifar_accel_0/interrupt, axi_dma_0/mm2s_introut} -> xlconcat -> axi_intc_0 -> IRQ_F2P
#  one clock FCLK_CLK0 = $::clk_mhz MHz, one reset domain (proc_sys_reset)
# Results (class = ap_return, logits[10]) are read over AXI4-Lite: no S2MM channel,
# no DDR weight traffic (all weights are on-chip ROMs inside the HLS IP).

if {![info exists ::clk_mhz]} { set ::clk_mhz 150 }
create_bd_design design_1

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} $ps
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $::clk_mhz \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {32} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} CONFIG.PCW_IRQ_F2P_INTR {1} \
] $ps

set clk [get_bd_pins processing_system7_0/FCLK_CLK0]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0
connect_bd_net $clk [get_bd_pins rst_ps7_0/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0/ext_reset_in]
set rstn [get_bd_pins rst_ps7_0/peripheral_aresetn]

create_bd_cell -type ip -vlnv xilinx.com:hls:cifar_accel:1.0 cifar_accel_0

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {14} \
    CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {0} \
    CONFIG.c_m_axi_mm2s_data_width {32} CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {64} \
] $dma

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0
set_property CONFIG.C_IRQ_CONNECTION {1} [get_bd_cells axi_intc_0]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0
set_property CONFIG.NUM_PORTS {2} [get_bd_cells xlconcat_0]

set c [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] $c
set c [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_hp0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $c

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins cifar_accel_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M02_AXI] [get_bd_intf_pins axi_intc_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins smc_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_hp0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins cifar_accel_0/img]

foreach p {
    processing_system7_0/M_AXI_GP0_ACLK processing_system7_0/S_AXI_HP0_ACLK
    cifar_accel_0/ap_clk axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk
    axi_intc_0/s_axi_aclk smc_ctrl/aclk smc_hp0/aclk
} { connect_bd_net $clk [get_bd_pins $p] }
foreach p {
    cifar_accel_0/ap_rst_n axi_dma_0/axi_resetn axi_intc_0/s_axi_aresetn smc_ctrl/aresetn smc_hp0/aresetn
} { connect_bd_net $rstn [get_bd_pins $p] }

connect_bd_net [get_bd_pins cifar_accel_0/interrupt] [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut]  [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins axi_intc_0/intr]
connect_bd_net [get_bd_pins axi_intc_0/irq]  [get_bd_pins processing_system7_0/IRQ_F2P]

proc place_slave {intf_pin offset} {
    set seg [get_bd_addr_segs -of_objects [get_bd_intf_pins $intf_pin]]
    assign_bd_address -target_address_space /processing_system7_0/Data $seg -offset $offset -range 64K -force
}
place_slave cifar_accel_0/s_axi_control 0x40000000
place_slave axi_dma_0/S_AXI_LITE        0x40400000
place_slave axi_intc_0/s_axi            0x41800000
assign_bd_address

regenerate_bd_layout
validate_bd_design
save_bd_design
set bd_file [get_files design_1.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "INFO: block design ready (FCLK0 = $::clk_mhz MHz). Next: vivado -mode batch -source vivado/build.tcl"
