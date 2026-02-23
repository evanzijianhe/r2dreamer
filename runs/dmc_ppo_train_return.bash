#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
SEED_END=400
SEED_STEP=100
MODAL=proprio # vision/proprio
METHOD=ppo # r2dreamer/ppo

# ==== Tasks ====
tasks=(
    dmc_acrobot_swingup
    dmc_ball_in_cup_catch
    dmc_cartpole_balance
    dmc_cartpole_balance_sparse
    dmc_cartpole_swingup
    dmc_cartpole_swingup_sparse
    dmc_cheetah_run
    dmc_finger_spin
    dmc_finger_turn_easy
    dmc_finger_turn_hard
    dmc_hopper_hop
    dmc_hopper_stand
    dmc_pendulum_swingup
    dmc_quadruped_run
    dmc_quadruped_walk
    dmc_reacher_easy
    dmc_reacher_hard
    dmc_walker_run
    dmc_walker_stand
    dmc_walker_walk
)

# ==== Loop ====
# ==== Loop ====
for task in "${tasks[@]}"
do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        python evals/plot_training_progress.py \
            --task ${task#dmc_} \
            --series ppo=logdir/ppo \
            --series-tag ppo=charts/episodic_return \
            --out logdir/plots/ppo_${task#dmc_}_return.png
    done
done

