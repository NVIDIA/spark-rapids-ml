# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# utility functions for mapping cluster name to id and running benchmark scripts
if [[ -z $DB_PROFILE ]]; then
    echo "please export DB_PROFILE per README.md"
    exit 1
fi

if [[ -z $BENCHMARK_HOME ]]; then
    echo "please export BENCHMARK_HOME per README.md"
    exit 1
fi

if [[ -z $WS_BENCHMARK_HOME ]]; then
    echo "please expoert WS_BENCHMARK_HOME per README.md"
    exit 1
fi

create_cluster() {
    cluster_type=$1

    cluster_name=spark-rapids-ml-$cluster_type-$USER

    # delete cluster with selected name if already created
    cluster_record=$( databricks clusters list --profile $DB_PROFILE | grep $cluster_name )
    if [ -n "$cluster_record" ]
    then
        cluster_id=`echo $cluster_record | cut -d ' ' -f 1`
        echo "found cluster named $cluster_name and id $cluster_id deleting ..."
        databricks clusters delete $cluster_id --profile $DB_PROFILE
    fi
    
    INIT_SCRIPT_DIR="${WS_BENCHMARK_HOME}/init_scripts"

    # sourcing allows variable substitution (e.g. cluster name) into cluster json specs
    
    cluster_spec=`source ${cluster_type}_cluster_spec.sh`
    echo $cluster_spec

    create_response=$( databricks clusters create --no-wait --json "$cluster_spec" --profile $DB_PROFILE )

    cluster_id=$( echo $create_response | grep cluster_id | cut -d \" -f 4 )
    if [ -z $cluster_id ]
    then
        echo "error creating cluster:"
        echo $create_response
        exit 1
    fi
    sleep 2
    cluster_status=""
    echo "waiting for cluster ${cluster_id} to start"
    while [[ $cluster_status != \"RUNNING\" ]]; do
        sleep 10
        cluster_status=$( databricks clusters get $cluster_id --profile $DB_PROFILE | grep -o \"RUNNING\" )
        echo -n "."
    done
}

get_run_status() {
    status=$( databricks jobs get-run $run_id --profile ${DB_PROFILE} | grep life_cycle_state | grep -o '\(TERMINATED\|ERROR\)' | head -n 1 )
}

get_task_run_id() {
    task_run_id=$( databricks jobs get-run $run_id --profile ${DB_PROFILE} | grep run_id | grep -o '\([0-9]*\)' | tail -n 1 )
}


run_bm() {

    algorithm=$1

    params_delimited=$( echo $@ | sed -e 's/^/"/g' | sed -e 's/$/"/g' | sed -e 's/ /", "/g' )

json_string=`cat <<EOF
{
    "run_name": "$algorithm",
    "tasks": [
        {
            "task_key": "run_task",
            "existing_cluster_id": "${cluster_id}",
            "spark_python_task": {
                "python_file": "dbfs:${BENCHMARK_HOME}/scripts/benchmark_runner.py",
                "parameters": [
                    ${params_delimited},
                    "--num_gpus",
                    "${num_gpus}",
                    "--num_cpus",
                    "${num_cpus}",
                    "--no_shutdown"
                ]
            }
        }
    ]               
}
EOF
`
    # set time limit for run, in seconds. run canceled after exceeding this
    if [ -z $TIME_LIMIT ]
    then
    TIME_LIMIT=3600
    fi

    submit_response=$( databricks jobs submit --no-wait --json "$json_string" --profile ${DB_PROFILE} )
    echo ${submit_response} | grep run_id > /dev/null
    if [[ $? != 0 ]]; then
        echo "error submitting run"
        echo "response:"
        echo $submit_response
        echo "request:"
        echo $json_string
        exit 1
    fi
    run_id=$( echo $submit_response | grep run_id | sed -e 's/[^0-9]*\([0-9]*\)[^0-9]*/\1/g' )
    echo "waiting for run $run_id to finish"
    sleep 5
    get_run_status
    duration=0
    while [[ $status != "TERMINATED" ]] && [[ $status != "ERROR" ]]; do
        echo -n "."
        if [[ $TIME_LIMIT != "" ]] && (( $duration > $TIME_LIMIT ))
        then
            echo "\ntime limit of $TIME_LIMIT minutes exceeded, canceling run"
            databricks jobs cancel-run $run_id --profile $DB_PROFILE
        fi
        sleep 10
        duration=$(( $duration + 10 ))
        get_run_status
    done

    get_task_run_id
    databricks jobs get-run-output $task_run_id --profile ${DB_PROFILE}
    if [[ $status == "ERROR" ]]; then
        echo "An error was encountered during run.  Exiting."
        exit 1
    else
        echo "done"
    fi
}
