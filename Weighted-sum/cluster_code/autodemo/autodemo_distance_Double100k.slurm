#!/bin/sh

## Resource Request
#SBATCH -J autodemo
#SBATCH -p Epyc7452
#SBATCH -q long
#SBATCH -N 1  #This specify the number of nodes you want to use, force N=1 because we have issues if irace is running on more than one node
#SBATCH -n 16
#SBATCH -o ./irace-%A.stdout
#SBATCH -e ./irace-%A.stderr
#SBATCH -D /home/jszpirer/TuttiFrutti/autodemo/irace/

## Job Steps
MISSION="$1"
ITER=$2
NBMISSIONS="$3"

source /home/jszpirer/env_project/bin/activate
source /home/jszpirer/setup-argos3.sh
export PATH
export OMPI_MCA_plm_rsh_disable_qrsh=1
export MPIRUN=/opt/ohpc/pub/mpi/openmpi4-gnu9/4.0.5/bin/mpirun

cd $MISSION
echo "$MISSION"
echo "Prepare Json file"

python3 /home/jszpirer/TuttiFrutti/autodemo/demoProcess_TuttiFrutti_distance_Double.py $MISSION $NBMISSIONS

echo "Begin irace iterations"

for ((i = 1; i <= ITER; i++))
do
  echo "iteration $i"
  $MPIRUN -x OMPI_MCA_plm_rsh_disable_qrsh -x PATH -v -n 1 $IRACE_HOME/bin/irace --exec-dir=results-folder --parallel 15 --mpi 1 --scenario scenarios/scenario100k.txt
  echo "Irace finished"
  echo "reading fsm.txt"
  Rscript fsm_collector.R
  echo "fsm_collector finished"
  python3 /home/jszpirer/TuttiFrutti/autodemo/pfsmCollector_TuttiFrutti_distance_Double.py $MISSION $i $NBMISSIONS
done

echo "end of experience"
