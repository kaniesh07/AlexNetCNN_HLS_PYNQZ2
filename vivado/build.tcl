# build.tcl -- synthesis, low-power implementation, timing check, bitstream, PYNQ files
#   vivado -mode batch -source vivado/build.tcl
# Output: build/overlay/cifar_hls.{bit,hwh} + xcifar_accel_hw.h (HLS register map)
#         build/reports/{utilization,utilization_hier,timing_summary,power}.rpt

set root    [file normalize [file join [file dirname [info script]] ..]]
set out_dir [file join $root build overlay]
set rpt_dir [file join $root build reports]
file mkdir $out_dir $rpt_dir
if {[catch {current_project}]} { open_project [file join $root build vivado cifar_hls_pynq.xpr] }
set jobs 4
if {[info exists ::env(JOBS)]} { set jobs $::env(JOBS) }

set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE PerformanceOptimized [get_runs synth_1]
set_property STEPS.POWER_OPT_DESIGN.IS_ENABLED true                 [get_runs impl_1]  ;# BRAM/FF clock gating
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore              [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true                  [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore              [get_runs impl_1]

reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synthesis failed" }
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed" }

open_run impl_1
report_utilization                  -file [file join $rpt_dir utilization.rpt]
report_utilization -hierarchical    -file [file join $rpt_dir utilization_hier.rpt]
report_timing_summary -max_paths 20 -file [file join $rpt_dir timing_summary.rpt]
report_power                        -file [file join $rpt_dir power.rpt]
set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
puts "TIMING: WNS = $wns ns, WHS = $whs ns"
if {$wns < 0 || $whs < 0} {
    puts "ERROR: timing not met -- rebuild HLS and Vivado with CLK_MHZ=100"
    exit 2
}

set bit [lindex [glob [file join $root build vivado cifar_hls_pynq.runs impl_1 *.bit]] 0]
set hwh [lindex [glob [file join $root build vivado cifar_hls_pynq.gen sources_1 bd design_1 hw_handoff design_1.hwh]] 0]
file copy -force $bit [file join $out_dir cifar_hls.bit]
file copy -force $hwh [file join $out_dir cifar_hls.hwh]
set hdr [file join $root hls cifar_hls solution1 impl misc drivers cifar_accel_v1_0 src xcifar_accel_hw.h]
if {[file exists $hdr]} { file copy -force $hdr [file join $out_dir xcifar_accel_hw.h] }
puts "DONE: overlay files in $out_dir"
