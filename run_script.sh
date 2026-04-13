#!/usr/bin/env bash


# Name of conda environment you want to activate
CONDA_ENV="biasens"

# A list/array of experiments (commands) you want to run
# Each command is a single string that includes your entire pipeline,
# e.g. activate environment
# Writing all experiments for BFFHQ 
declare -a EXPERIMENTS=(



    "CUDA_VISIBLE_DEVICES=0 python train.py --train_lff_unaware --dataset=cmnist --percent=5pct --lr=0.01 --exp=lff_baseline_5pct --fix_randomseed --seed=42 |& tee exp_results/cmnist/5/lff.log"


)
# FORTO lff  done after FORTO _lff _SAVED 
# Loop over each experiment and launch in a new 'screen'
for i in "${!EXPERIMENTS[@]}"; do
  # you can generate session name from the experiment index or any naming pattern
  SESSION_NAME="experiments_$i"

  echo "Launching screen session: $SESSION_NAME"

  # The -dmS options mean: 
  #  -d = start screen in 'detached' mode 
  #  -m = ignore $STY environment variable (force new screen)
  #  -S <session_name> = give the screen session a name
  # We feb a bash -c "..." command which:
  #   1) sources conda
  #   2) activates the environment
  #   3) febs the experiment
  #   4) (optionally) keeps the screen open or closes after the script ends
  screen -dmS "$SESSION_NAME" bash -c "source $CONDA_SETUP; conda activate $CONDA_ENV; ${EXPERIMENTS[$i]}"

done
