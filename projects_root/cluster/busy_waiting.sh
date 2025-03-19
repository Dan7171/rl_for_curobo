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
# USERNAME=evrond
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
alloc_cpus=$(echo "$job_info" | grep -oP 'NumCPUs=\K\S+')
alloc_mem=$(echo "$job_info" | grep -oP 'AllocTRES=.*mem=\K[^,]+')
alloc_gpu=$(echo "$job_info" | grep -oP 'gres/gpu=\K[0-9]+')
alloc_gpu_type=$(echo "$job_info" | grep -oP 'gres/gpu:\K[^,]+')

# Function to validate resources
validate_resources() {
    local valid=true
    
    # CPU validation
    if ! [[ "$alloc_cpus" =~ ^[0-9]+$ ]] || [ "$alloc_cpus" -ne "$CPU" ]; then
        echo "Allocated CPU count ($alloc_cpus) does not match requested ($CPU)"
        valid=false
    fi
    
    # Memory validation
    if [ "$alloc_mem" != "$MEM" ]; then
        echo "Allocated memory ($alloc_mem) does not match requested ($MEM)"
        valid=false
    fi
    
    # GPU count validation
    if ! [[ "$alloc_gpu" =~ ^[0-9]+$ ]] || [ "$alloc_gpu" -ne "$GPU" ]; then
        echo "Allocated GPU count ($alloc_gpu) does not match requested ($GPU)"
        valid=false
    fi
    
    # GPU type validation
    if [ "$alloc_gpu_type" != "rtx_3090" ]; then
        echo "Allocated GPU type ($alloc_gpu_type) does not match requested (rtx_3090)"
        valid=false
    fi
    
    echo "$valid"
}

# Check resources and retry if needed
while true; do
    validation_result=$(validate_resources)
    if [ "$validation_result" = "true" ]; then
        break
    else
        echo "Resource allocation mismatch. Cancelling current job and retrying..."
        scancel $JOBID
        rm -f interactive.sbatch
        sleep 5  # Wait a bit before retrying
        
        # Create and submit new job
        create_sbatch > interactive.sbatch
        export JOBID=$(sbatch interactive.sbatch | awk '{print $4}')
        
        if [ -z "${JOBID}" ]; then
            echo "Failed to submit new job. Exiting."
            exit 1
        fi
        
        # Wait for job to start
        NODE=$(squeue -hj $JOBID -O nodelist)
        while [[ -z "${NODE// }" ]]; do
            echo -n "."
            sleep 3
            NODE=$(squeue -hj $JOBID -O nodelist)
        done
        
        # Get updated job info
        job_info=$(scontrol show job $JOBID)
        alloc_cpus=$(echo "$job_info" | grep -oP 'NumCPUs=\K\S+')
        alloc_mem=$(echo "$job_info" | grep -oP 'AllocTRES=.*mem=\K[^,]+')
        alloc_gpu=$(echo "$job_info" | grep -oP 'gres/gpu=\K[0-9]+')
        alloc_gpu_type=$(echo "$job_info" | grep -oP 'gres/gpu:\K[^,]+')
    fi
done

echo "Resource allocation validated successfully!"

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
echo "    ${USER}@${HOST_IP}"
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
