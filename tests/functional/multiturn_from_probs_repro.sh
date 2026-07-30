#!/bin/bash
# Repro harness for the Megatron_4 multiturn token_mult_prob_error regression (T1: joint
# flashinfer kernel variant). Applies a one-line MLM patch routing the joint top-k/top-p
# sampling branch through *_from_probs (the pre-#5791 kernel family, as at the last green
# MLM pin cf2f07d7) while keeping the new dispatch/eagerness/determinism, then runs the
# failing multiturn config (top_p=0.9, top_k=8000). Extra Hydra overrides pass through, e.g.:
#   bash tests/functional/multiturn_from_probs_repro.sh \
#       policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=false
# GREEN here + RED on grpo_megatron_generation_multiturn.sh (unpatched) = T1 confirmed.
set -eou pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
MLM=$PROJECT_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
PATCH=$PROJECT_ROOT/tools/patches/mlm-joint-sampling-from-probs.patch

if git -C "$MLM" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[repro] MLM from_probs patch already applied"
else
    git -C "$MLM" apply "$PATCH"
    echo "[repro] applied MLM from_probs patch to $MLM"
fi

exec bash "$SCRIPT_DIR/grpo_megatron_generation_multiturn.sh" "$@"
