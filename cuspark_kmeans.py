from pyspark.sql import SparkSession
import time
from pyspark import BarrierTaskContext
from pyspark.sql.types import StructType
from pyspark.sql.types import ArrayType 
from pyspark.sql.types import DoubleType
from pyspark import RDD
from pyspark.sql import DataFrame
import cupy as cp

from cuml.cluster.kmeans_mg import KMeansMG as CumlKmeans
from raft.dask.common.nccl import nccl
from raft.dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft.common import Handle

class CuPCA:
    def __init__(self):
        self.ncclUniqueId = nccl.get_unique_id()
        print(type(self.ncclUniqueId))
        print(self.ncclUniqueId)
        print("------------end self.ncclUniqueId--------")

    def fit(self, df: DataFrame):

        sparkCtx = SparkSession.builder.getOrCreate()
        uniqueId = sparkCtx.sparkContext.broadcast(self.ncclUniqueId)

        def toyFit(iterator):

            print(type(uniqueId.value))
            print(uniqueId.value)
            print("--------uniqueId.value------")
            nWorkers = 1
            wid = 0
            ncclComm = nccl()
            ncclComm.init(nWorkers, uniqueId.value, wid)
            handle = Handle(n_streams = 0)
            inject_comms_on_handle_coll_only(handle, ncclComm, nWorkers, wid, True)

            kwargs = {'init' : 'k-means||', 'n_clusters' : 2, 'random_state' : 100}
            kmeansObject = CumlKmeans(handle=handle, output_type = 'cupy', **kwargs)
            print(iterator)
            print("---------iterator----------")
            cupyArraysList = [cp.array(list(iterator))]
            print(cupyArraysList)
            print("----cupyArraysList[0]----")

            kmeansObject = kmeansObject.fit(cupyArraysList[0], sample_weight = None)


            res = kmeansObject.predict(cupyArraysList[0])
            print(type(res))
            print(res)
            print("-------end res----------")
            yield kmeansObject 

        self.modelRDD = df.rdd.map(lambda row : row[self.inputCol]).mapPartitions(toyFit).cache()
        print(self.modelRDD.count())
        print("----------in fit function------")
        return self

    def setInputCol(self, inputCol = "feature"):
        self.inputCol = inputCol
        return self

    def setOutputCol(self, outputCol = "pca_feature"):
        self.outputCol = outputCol
        return self

    def setK(self, topk):
        self.topK = topk
        return self

if __name__ == "__main__":
    data = [[7.02066012, -9.14564486], 
            [6.94755379, -9.18329509], 
            [-2.6244721,   2.96211854], 
            [ 6.9954565,  -9.25203817], 
            [ 6.97064513, -9.39882037], 
            [ 6.82424233, -9.10935929], 
            [-2.6425227,   3.02642711], 
            [-2.74728146,  3.04815685]] 

    numPart = 1
    topk = 1
    spark = SparkSession.builder.master("local[" + str(numPart) + "]").getOrCreate()
    #spark = SparkSession.builder \
    #        .master("spark://jinfengl-dt:7077") \
    #        .config("spark.submit.pyFiles", "/home/jinfengl/project/nvidia/pyspark-pca/sparkcomm.py") \
    #        .getOrCreate()

    t_start = time.time()
    rdd = spark.sparkContext.parallelize(data, numPart).map(lambda row : (row, ))

    df = rdd.toDF(["feature"]).cache()
    df.count()

    t_load = time.time()
    print("load time is " + str(t_load - t_start))

    gpuPca = CuPCA().setInputCol("feature").setOutputCol("pca_features").setK(topk)

    t_fit_start = time.time()
    gpuModel = gpuPca.fit(df)

    t_fit = time.time()
    print("fit time is " + str(t_fit - t_fit_start))

    #gpuModel.transform(df).count() #show(truncate=False)
    #t_transform = time.time()
    #print("transform time is " + str(t_transform - t_fit))
  
    print("cuML on Spark PCA finishes")