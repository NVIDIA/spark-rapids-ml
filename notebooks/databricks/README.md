To run notebooks on Databricks do the following:
- If you don't already have it, install [databricks cli](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/) and create and save an [access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) to your workspace using the workspace UI. 
- Create a zip file of `sparkcuml` directory via `zip -r sparkcuml.zip sparkcuml` command at the top level of the repo and copy to a location in dbfs using the databricks cli command `databricks fs cp sparkcuml.zip <dbfs:/dbfs location>` 
- Edit the file [init-pip-11.8.sh](init-pip-11.8.sh) to set the `SPARKCUML_ZIP` variable to the dbfs location used above and upload the resulting modifed file to dbfs using the 
- Create a cluster using Databricks 10.4LTS runtime (11.3 is not yet compatible with spark-rapids) using at least 2 gpu based workers and add the init file dbfs location used above to the spark-config section of the cluster.
- Select your workspace and upload desired notebook via `Import` in the drop down menu for your workspace.
