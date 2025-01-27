## Running notebooks on Dataproc

If you already have a Dataproc account, you can run the example notebooks on a Dataproc cluster, as follows:
- Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and initialize it via `gcloud init`.
- Configure the following settings:
  ```
  export PROJECT=<your_project>
  export DATAPROC_REGION=<your_dataproc_region>
  export COMPUTE_REGION=<your_compute_region>
  export COMPUTE_ZONE=<your_compute_zone>

  gcloud config set project ${PROJECT}
  gcloud config set dataproc/region ${DATAPROC_REGION}
  gcloud config set compute/region ${COMPUTE_REGION}
  gcloud config set compute/zone ${COMPUTE_ZONE}
  ```
- Create a GCS bucket if you don't already have one:
  ```
  export GCS_BUCKET=<your_gcs_bucket_name>

  gcloud storage buckets create gs://${GCS_BUCKET}
  ```
- Upload the initialization scripts to your GCS bucket:
  ```
  gsutil cp spark_rapids_ml.sh gs://${GCS_BUCKET}
  curl -LO https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/spark-rapids/spark-rapids.sh
  gsutil cp spark-rapids.sh gs://${GCS_BUCKET}/spark-rapids.sh
  ```
- Create a cluster with at least two single-gpu workers.  **Note**: in addition to the initialization script from above, this also uses the standard [initialization actions](https://github.com/GoogleCloudDataproc/initialization-actions) for installing the GPU drivers and RAPIDS.
  
  If you wish to enable [no-import-change](../README.md#no-import-change) UX for the cluster, change the `spark-rapids-ml-no-import-enabled` metadata value to `1` in the command.  The initialization script `spark_rapids_ml.sh` checks this metadata value and modifies the run time accordingly.

  ```
  export RAPIDS_VERSION=24.12.0

  gcloud dataproc clusters create $USER-spark-rapids-ml \
  --image-version=2.2-ubuntu22 \
  --region ${COMPUTE_REGION} \
  --master-machine-type n1-standard-16 \
  --master-accelerator type=nvidia-tesla-t4,count=1 \
  --num-workers 2 \
  --worker-min-cpu-platform=Intel\ Skylake \
  --worker-accelerator type=nvidia-tesla-t4,count=1 \
  --worker-machine-type n1-standard-16 \
  --num-worker-local-ssds 4 \
  --worker-local-ssd-interface=NVME \
  --initialization-actions gs://${GCS_BUCKET}/spark-rapids.sh,gs://${GCS_BUCKET}/spark_rapids_ml.sh \
  --initialization-action-timeout=20m \
  --optional-components=JUPYTER \
  --metadata gpu-driver-provider="NVIDIA" \
  --metadata rapids-runtime=SPARK \
  --metadata rapids-version=${RAPIDS_VERSION} \
  --metadata spark-rapids-ml-no-import-enabled=0 \
  --bucket ${GCS_BUCKET} \
  --enable-component-gateway \
  --subnet=default \
  --no-shielded-secure-boot
  ```
- In the [Dataproc console](https://console.cloud.google.com/dataproc/clusters), select your cluster, go to the "Web Interfaces" tab, and click on the "JupyterLab" link.
- In JupyterLab, upload the desired [notebook](../) via the `Upload Files` button.  For the no-import-change UX, you can try the example [kmeans-no-import-change.ipynb](../kmeans-no-import-change.ipynb).
- Add the following to a new cell at the beginning of the notebook, since Dataproc does not start the `SparkSession` by default:
  ```
  from pyspark.sql import SparkSession

  spark = SparkSession.builder \
  .appName("spark-rapids-ml") \
  .config("spark.executor.resource.gpu.amount", "1") \
  .config("spark.task.resource.gpu.amount", "1") \
  .config("spark.executorEnv.CUPY_CACHE_DIR", "/tmp/.cupy") \
  .config("spark.locality.wait", "0") \
  .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
  .config("spark.sql.execution.arrow.maxRecordsPerBatch", "100000") \
  .config("spark.rapids.memory.gpu.pooling.enabled", "false") \
  .config("spark.rapids.memory.gpu.reserve", "90") \
  .getOrCreate()
  ```
  **Note**: these settings are for demonstration purposes only.  Additional tuning may be required for optimal performance.
- Run the notebook cells.  **Note**: you may need to change file paths to use `hdfs://` paths.
- Add the following to a new cell at the end of the notebook to close the `SparkSession`:
  ```
  spark.stop()
  ```