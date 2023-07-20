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

#executable and redshift list
MT4EXEC=./emu.exe
NZ=25
Z_LIST=( 2.02 1.799 1.61 1.376 1.209 1.006 0.779 0.736 0.695 0.656 0.6431 0.618 0.578 0.539 0.502 0.471 0.434 0.402 0.364 0.304 0.242 0.212 0.154 0.101 0.0 )

#read inputs
OMH2=${1}
OBH2=${2}
SIG8=${3}
HRED=${4}
NSCA=${5}
WDE0=${6}
WDEA=${7}
ONH2=${8}

#generate emu parameter file and run
rm xstar.dat EMU[0-9]*.txt MT4_emu_output.dat >&/dev/null

for ((I=0; I<${NZ}; I++)); do
  echo ${OMH2} ${OBH2} ${SIG8} ${HRED} ${NSCA} \
	${WDE0} ${WDEA} ${ONH2} ${Z_LIST[${I}]} >> xstar.dat
done

${MT4EXEC}

#process outputs
PRE=$(echo ${HRED} | awk '{print $1*$1*$1}')

for ((I=0; I<${NZ}; I++)); do
  PCORR=$(echo ${PRE} ${Z_LIST[${I}]} | awk '{print $1*(1.0+$2)*(1.0+$2)}')
  cat EMU${I}.txt \
	| awk -v h=${HRED} -v pc=${PCORR} -v z=${Z_LIST[${I}]} \
	'{print -log(1.0+z), log($1/h), log($2*pc)}' \
	>> MT4_emu_output.dat
done

rm EMU[0-9]*.txt

