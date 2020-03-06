#!/bin/bash -e

cd $(dirname $0)

task_file=$1
[ -z "$task_file" ] && task_file=attack_tasks.txt
[ -z "$JOBS" ] && JOBS=4

export ARGV1_NEED_PARSE=1

exec parallel -j$JOBS --lb --termseq INT,500,INT,500,INT,2000,KILL,25 \
    "./step2_attack.sh {}" \
    :::: $task_file
