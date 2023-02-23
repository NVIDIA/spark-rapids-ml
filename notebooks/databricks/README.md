To run notebooks on Databricks do the following:
- If you don't already have it, install [databricks cli](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/) and create and save an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) to your workspace using the workspace UI. 
- Inside the `src` directory, create a zip file of the `spark_rapids_ml` directory via `zip -r spark_rapids_ml.zip spark_rapids_ml` command at the top level of the repo and copy to a location in dbfs using the databricks cli command `databricks fs cp spark_rapids_ml.zip <dbfs:/dbfs location>` 
- Edit the file [init-pip-cuda-11.8.sh](init-pip-cuda-11.8.sh) to set the `SPARK_RAPIDS_ML_ZIP` variable to the dbfs location used above and upload the resulting modifed `.sh` file to dbfs using the databricks cli.  Note the path here starts with `/dbfs/` which is where `dbfs` filesystem is mounted in the databricks runtime containers. 
- Create a cluster using Databricks 10.4LTS runtime (11.3 is not yet compatible with spark-rapids) using at least 2 gpu based workers and add the init file dbfs location used above to the spark-config section of the cluster and using the Spark configs below:
```
```
- Select your workspace and upload desired notebook via `Import` in the drop down menu for your workspace.
