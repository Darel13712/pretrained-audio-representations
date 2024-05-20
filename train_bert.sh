#!/bin/bash

# Default values
time_limit_hours=12
#cpus_per_task=40
memory_gb=20
sample_type="item"
model="musicnn"
maxseqlen=200
batch=10000
epochs=100
item_freeze=1
user_freeze=0
logdir=""
last_epoch=""
comment="nc"
user_init=1
item_dynamic_unfreeze=0
hidden_dim=128
use_confidence=0
l2=0
#k=100

# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --time) time_limit_hours="$2"; shift ;;
#    --cpus-per-task) cpus_per_task="$2"; shift ;;
    --memory) memory_gb="$2"; shift ;;
    --sample) sample_type="$2"; shift ;;
    --model) model="$2"; shift ;;
    --maxseqlen) maxseqlen="$2"; shift ;;
    --batch) batch="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --ifreeze) item_freeze="$2"; shift ;;
    --ufreeze) user_freeze="$2"; shift ;;
    --logdir) logdir="$2"; shift ;;
    --lastepoch) last_epoch="$2"; shift ;;
    --comment) comment="$2"; shift ;;
    --uinit) user_init="$2"; shift ;;
    --iunfreeze) item_dynamic_unfreeze="$2"; shift ;;
    --hiddim) hidden_dim="$2"; shift ;;
    --useconf) use_confidence="$2"; shift ;;
    --l2) l2="$2"; shift ;;
#    --k) k="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Construct job name
job_name="bert_${model}_${hidden_dim}_${maxseqlen}_${comment}"

# Format time limit as a string
time_limit="${time_limit_hours}:00:00"

cmd="python ./train_bert.py $model $sample_type $maxseqlen $batch $epochs $item_freeze $user_freeze $comment $user_init $item_dynamic_unfreeze $hidden_dim $use_confidence $l2"
[ -n "$logdir" ] && cmd+=" $logdir"  # Append logdir if it's not empty
[ -n "$last_epoch" ] && cmd+=" $last_epoch"

# Generate Slurm script as a string
slurm_script=$(cat << EOM
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --job-name="$job_name"
#SBATCH --time=$time_limit
#SBATCH --mem=${memory_gb}G
#SBATCH --output=logs/${job_name}.out


source ~/.e/bin/activate
module load cudnn

$cmd
EOM
)

# Submit the Slurm job by piping the script to sbatch
echo "$slurm_script" | sbatch