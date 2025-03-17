#!/bin/bash

# Help message
print_help() {
    echo
    echo "Usage: sinteractive [OPTIONS]"
    echo
    echo "Available arguments:"
    echo "  --part    Slurm partition.            
    echo "  --qos     Quality Of Service.      
    echo "  --time    Running time limit.          
    echo "  --cpu     Number of cpu cores.         
    echo "  --mem     Amount of RAM memory (GB).   
    echo "  --gpu     Number of GPU cards.         
    echo

    exit 1
}

if [[ "$#" -eq 0 ]]; then
    echo ""
    echo "No arguments provided. Using defaults."
    echo "Run 'sinteractive --help' to check for available options."
fi

# Default values
PART=rtx_3090 
QOS=normal
TIME="12:00:00"
CPU=4
MEM="8G"
GPU=1

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --part)
            PART=$2
            shift 2
            ;;
        --qos)
            QOS=$2
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpu)
            CPU=$2
            shift 2
            ;;
        --mem)
            MEM="$2G"
            shift 2
            ;;
        --gpu)
            GPU=$2
            shift 2
            ;;
        --help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

# Output the configuration
echo ""
echo "Configuration:"
echo "  Partition:        $PART"
echo "  QoS:              $QOS"
echo "  Time Limit:       $TIME"
echo "  CPU Cores:        $CPU"
echo "  RAM Memory:       $MEM"
echo "  GPU Cards:        $GPU"
echo ""

########################################
#
# TRAP SIGINT AND SIGTERM OF THIS SCRIPT
function control_c {
    echo -en "\n SIGINT: TERMINATING SLURM JOBID $JOBID AND EXITING \n"
    scancel $JOBID
    rm interactive.sbatch
    exit $?
}
trap control_c SIGINT
trap control_c SIGTERM
#
# SBATCH FILE FOR ALLOCATING COMPUTE RESOURCES TO RUN NOTEBOOK SERVER

create_sbatch() {
cat << EOF
#!/bin/bash
#
#SBATCH --job-name interactive
#SBATCH --partition $PART
#SBATCH --qos $QOS
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --time=$TIME
#SBATCH --gpus=$GPU
#SBATCH --mem=$MEM
##SBATCH -J interactive
#SBATCH -o \$CWD/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

#
echo date
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  \$launch "
#module load cuda/11.1
while true; do
        eval \$launch
done
EOF
}
#
## CREATE INTERACTIVE SBATCH FILE
export CWD=$(pwd)

create_sbatch > interactive.sbatch

#
# START NOTEBOOK SERVER
#
export JOBID=$(sbatch interactive.sbatch  | awk '{print $4}')

if [ -z "${JOBID}" ]; then
  exit 0
fi

NODE=$(squeue -hj $JOBID -O nodelist )
if [[ -z "${NODE// }" ]]; then
   echo  " "
   echo -n "    WAITING FOR RESOURCES TO BECOME AVAILABLE (CTRL-C TO EXIT) ..."
fi
while [[ -z "${NODE// }" ]]; do
   echo -n "."
   sleep 3
   NODE=$(squeue -hj $JOBID -O nodelist )
done
HOST_NAME=$(squeue -j $JOBID -h -o  %B)
HOST_IP=$(grep -i $HOST_NAME /etc/hosts | awk '{ print $1 }')
TIMELIM=$(squeue -hj $JOBID -O timeleft )

########################################
# QUERY ALLOCATED RESOURCES AND VALIDATE REQUEST
echo ""
echo "Querying allocated resources..."
job_info=$(scontrol show job $JOBID)
echo "$job_info"

# Extract allocated partition, CPUs, memory, and GPU count
# alloc_partition=$(echo "$job_info" | grep -oP 'Partition=\K\S+')
alloc_cpus=$(echo "$job_info" | grep -oP 'NumCPUs=\K\S+')
alloc_mem=$(echo "$job_info" | grep -oP 'AllocTRES=.*mem=\K[^,]+')
alloc_gpu=$(echo "$job_info" | grep -oP 'gres/gpu=\K[0-9]+')

# Check each requested resource; if any don't match, cancel the job and exit

# if [[ "$alloc_partition" != "$PART" ]]; then
#     echo "Allocated partition ($alloc_partition) does not match requested ($PART). Cancelling job."
#     scancel $JOBID
#     rm interactive.sbatch
#     exit 1
# fi

if [[ "$alloc_cpus" -ne "$CPU" ]]; then
    echo "Allocated CPU count ($alloc_cpus) does not match requested ($CPU). Cancelling job."
    scancel $JOBID
    rm interactive.sbatch
    exit 1
fi

if [[ "$alloc_mem" != "$MEM" ]]; then
    echo "Allocated memory ($alloc_mem) does not match requested ($MEM). Cancelling job."
    scancel $JOBID
    rm interactive.sbatch
    exit 1
fi

if [[ "$alloc_gpu" -ne "$GPU" ]]; then
    echo "Allocated GPU count ($alloc_gpu) does not match requested ($GPU). Cancelling job."
    scancel $JOBID
    rm interactive.sbatch
    exit 1
fi
#
########################################
#
# FORMAT remaining time for display
if [[ $TIMELIM == *"-"* ]]; then
  DAYS=$(echo $TIMELIM | awk -F '-' '{print $1}')
  HOURS=$(echo $TIMELIM | awk -F '-' '{print $2}' | awk -F ':' '{print $1}')
  MINS=$(echo $TIMELIM | awk -F ':' '{print $2}')
  TIMELEFT="THIS SESSION WILL TIMEOUT IN $DAYS DAY $HOURS HOUR(S) AND $MINS MINS "
else
  HOURS=$(echo $TIMELIM | awk -F ':' '{print $1}' )
  MINS=$(echo $TIMELIM | awk -F ':' '{print $2}')
  TIMELEFT="THIS SESSION WILL TIMEOUT IN $HOURS HOUR(S) AND $MINS MINS "
fi

echo " "
echo " "
echo "  --------------------------------------------------------------------"
echo "    INTERACTIVE SESSION STARTED ON NODE $NODE           "
echo "    $TIMELEFT"
echo "    SESSION LOG WILL BE STORED IN interactive_${JOBID}.out  "
echo "  --------------------------------------------------------------------"
echo "  "
echo "    TO ACCESS THIS COMPUTE NODE, USE THIS IN YOUR IDE: "
echo "  "
echo "    \$USER@${HOST_IP}  "
echo "  --------------------------------------------------------------------"
echo "  "
echo "    TO TERMINATE THIS SESSION ISSUE THE FOLLOWING COMMAND: "
echo "  "
echo "       scancel $JOBID "
echo "  --------------------------------------------------------------------"
echo "  "
#
# CLEANUP
rm interactive.sbatch
#
# EOF
