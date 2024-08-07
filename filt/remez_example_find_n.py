#! /usr/bin/env python3

from filtEst import *

sFs      = 4096
sFpb     = 63
sFsb     = 120
sApb     = 0.5
sAsb     = 40

sN = firFindOptN(sFs, sFpb, sFsb, sApb, sAsb)

(h, w, H, Rpb, Rsb, Hpb_min, Hpb_max, Hsb_max) =  fir_calc(sFs, sFpb, sFsb, sApb, sAsb, sN)

plotFrequResp(w, H, sFs, sFpb, sFsb, Hpb_min, Hpb_max, Hsb_max)
plt.tight_layout()
plt.savefig("remez_example_find_n.svg")



