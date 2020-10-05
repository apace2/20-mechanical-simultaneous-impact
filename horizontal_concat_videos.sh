#!/bin/bash

ffmpeg -i figs/RigidBiped-0.1.mp4 -i figs/RigidBiped0.mp4 -i figs/RigidBiped0.1.mp4 -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[vid]' -map [vid] -c:v libx264 -crf 22 -preset veryfast figs/RigidBiped.mp4
ffmpeg -i figs/PCrBiped-0.1.mp4 -i figs/PCrBiped0.mp4 -i figs/PCrBiped0.1.mp4 -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[vid]' -map [vid] -c:v libx264 -crf 22 -preset veryfast figs/PCrBiped.mp4
ffmpeg -i figs/DP-0.1.mp4 -i figs/DP0.mp4 -i figs/DP0.1.mp4 -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[vid]' -map [vid] -c:v libx264 -crf 22 -preset veryfast figs/DP.mp4
ffmpeg -i figs/DP_control-0.05.mp4 -i figs/DP_control0.mp4 -i figs/DP_control0.05.mp4 -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[vid]' -map [vid] -c:v libx264 -crf 22 -preset veryfast figs/DP_control.mp4
