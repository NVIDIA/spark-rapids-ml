# utility functions for mapping cluster name to id and running benchmark scripts
if [[ -z $DB_PROFILE ]]; then
    echo "please export DB_PROFILE per README.md"
    exit 1
fi

if [[ -z $BENCHMARK_HOME ]]; then
    echo "please export BENCHMARK_HOME per README.md"
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
        databricks clusters delete --cluster-id $cluster_id --profile $DB_PROFILE
    fi

    # sourcing allows variable substitution (e.g. cluster name) into cluster json specs
    cluster_spec=`source ${cluster_type}_cluster_spec.sh`
    echo $cluster_spec

    create_response=$( databricks clusters create --json "$cluster_spec" --profile $DB_PROFILE )

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
        cluster_status=$( databricks clusters get --cluster-id $cluster_id --profile $DB_PROFILE | grep -o \"RUNNING\" )
        echo -n "."
    done
}

get_run_status() {
    status=$( databricks runs get-output --run-id $run_id --profile ${DB_PROFILE} | grep life_cycle_state | grep -o '\(TERMINATED\|ERROR\)' | head -n 1 )
}

run_bm() {

    algorithm=$1

    params_delimited=$( echo $@ | sed -e 's/^/"/g' | sed -e 's/$/"/g' | sed -e 's/ /", "/g' )

json_string=`cat <<EOF
{
    "run_name": "$algorithm",
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
EOF
`
    # set time limit for run, in seconds. run canceled after exceeding this
    if [ -z $TIME_LIMIT ]
    then
    TIME_LIMIT=3600
    fi

    submit_response=$( databricks runs submit --json "$json_string" --profile ${DB_PROFILE} )
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
            databricks runs cancel --run-id $run_id --profile $DB_PROFILE
        fi
        sleep 10
        duration=$(( $duration + 10 ))
        get_run_status
    done

    databricks runs get-output --run-id $run_id --profile ${DB_PROFILE}
    if [[ $status == "ERROR" ]]; then
        echo "An error was encountered during run.  Exiting."
        exit 1
    else
        echo "done"
    fi
}
