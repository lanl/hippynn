#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --qos=long
#SBATCH -p ml4chem
#SBATCH --job-name=EXAFStrain

#This run is basically as Kipton suggested, shortened for testing.

source ~/ml4chem/envs/p38-parsl-ani.bash

#python Run-train.py /vast/home/bnebgen/scratch/Al-Rad-MD/MD-run-1323-3-0/md-1/ /vast/home/bnebgen/scratch/HIPNN-Al-3/HIPNN-Al-03-0/ /vast/home/bnebgen/scratch/HIPNN-Al-3/HIPNN-Al-03-0/ model-00 $runTemp

CURMOD=<path_to_DFT_model>
echo $CURMOD

for LOOPI in 01 02 03 04 05
do
	echo "starting loop ${LOOPI}"	
	
	#Args:              HIPNNmodel,      tmpDir,                logFile,     targTemp, targPres, trajFile
	export CUDA_VISIBLE_DEVICES=0
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-a/sub-0 md-${LOOPI}-a-0.log 1023 6.324211e-07 md-${LOOPI}-a-0.traj &
	export CUDA_VISIBLE_DEVICES=1    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-a/sub-1 md-${LOOPI}-a-1.log 1023 6.324211e-07 md-${LOOPI}-a-1.traj &
	export CUDA_VISIBLE_DEVICES=2    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-a/sub-2 md-${LOOPI}-a-2.log 1023 6.324211e-07 md-${LOOPI}-a-2.traj &
	export CUDA_VISIBLE_DEVICES=3    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-a/sub-3 md-${LOOPI}-a-3.log 1023 6.324211e-07 md-${LOOPI}-a-3.traj &
	export CUDA_VISIBLE_DEVICES=0    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-b/sub-0 md-${LOOPI}-b-0.log 1323 6.324211e-07 md-${LOOPI}-b-0.traj &
	export CUDA_VISIBLE_DEVICES=1    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-b/sub-1 md-${LOOPI}-b-1.log 1323 6.324211e-07 md-${LOOPI}-b-1.traj &
	export CUDA_VISIBLE_DEVICES=2    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-b/sub-2 md-${LOOPI}-b-2.log 1323 6.324211e-07 md-${LOOPI}-b-2.traj &
	export CUDA_VISIBLE_DEVICES=3    
	python Run-MD.py ${CURMOD} /ram/tmp/md-${LOOPI}-b/sub-3 md-${LOOPI}-b-3.log 1323 6.324211e-07 md-${LOOPI}-b-3.traj &
		
	wait
	#Inputs:            curModelDir,          DFTModelDir,                               newModelDir,                 mdDirStr,                      targTempStr
	python Run-train-2.py ${CURMOD} <path_to_DFT_model> model-${LOOPI} /ram/tmp/md-${LOOPI}-a/:/ram/tmp/md-${LOOPI}-b/ 1023:1323
	CURMOD=${SLURM_SUBMIT_DIR}/model-${LOOPI}
	echo ${CURMOD}
done
