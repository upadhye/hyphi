#!/bin/bash
##
##    Copyright 2023 Amol Upadhye
##
##    This file is part of hyphi.
##
##    hyphi is free software: you can redistribute
##    it and/or modify it under the terms of the GNU General
##    Public License as published by the Free Software
##    Foundation, either version 3 of the License, or (at
##    your option) any later version.
##
##    hyphi is distributed in the hope that it
##    will be useful, but WITHOUT ANY WARRANTY; without
##    even the implied warranty of MERCHANTABILITY or
##    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
##    Public License for more details.
##
##    You should have received a copy of the GNU General
##    Public License along with hyphi.  If not,
##    see <http://www.gnu.org/licenses/>.
##

#Usage notes: ee2.exe and ee2_bindata.dat must be copied or symlinked to the 
#working directory.  Then, hyphi must be called with A_s as an argument.

EE2EXEC=./ee2.exe
NZ=40
Z_LIST=( 10 8 6.7 5.7 4.6 4 3.5 3 2.7 2.4 2.1 1.9 1.8 1.6 1.4 1.3 1.2 1.1 1 .9 .8 .75 .69 .61 .56 .51 .44 .4 .36 .33 .28 .25 .21 .17 .14 .12 .08 .06 .04 0 )

#read inputs
OM=${1}
OB=${2}
AS=${3}
HR=${4}
NS=${5}
W0=${6}
WA=${7}
MN=${8}

#clean up files from previous runs
rm ee20.dat ee2_output.txt ee2_parfile.par >&/dev/null

#run Euclid Emulator 2
echo -n "${OB},${OM},${MN},${NS},${HR},${W0},${WA},${AS}" > ee2_parfile.par

for ((I=0; I<${NZ}; I++)); do
    echo -n ",${Z_LIST[${I}]}" >> ee2_parfile.par
done
echo >> ee2_parfile.par

${EE2EXEC} --outdir . --outfile ee2 -p ee2_parfile.par > ee2_output.txt

#arrange outputs into interp file: ln(a), ln(k), ln(P/P_lin)
rm EE2_emu_output.dat >&/dev/null
for ((I=0; I<${NZ}; I++)); do
    A=$(echo "1.0/(1.0+${Z_LIST[${I}]})" | bc -l)
    grep "^[^#]" ee20.dat | cut -f 1,$((${I}+2)) \
	| awk -v a=${A} '{print log(a),log($1),log($2)}' >> EE2_emu_output.dat
done
    
#clean up; comment out this line to keep these for diagnostic purposes
rm ee20.dat ee2_output.txt ee2_parfile.par
