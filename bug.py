from enum import unique
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import ArrayType 
from pyspark.sql.types import DoubleType
from pyspark import RDD
from pyspark.sql import DataFrame

import cupy as cp
from cuml.decomposition.pca_mg import PCAMG as CumlPCA

from raft.dask.common.nccl import nccl
from raft.dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft.common import Handle

import time

# borrow from Erik:torchdist_on_spark
def _context_info():
    from pyspark import BarrierTaskContext
    context = BarrierTaskContext.get()
    tasks = context.getTaskInfos()
    return (tasks[0].address.split(":")[0], context.partitionId(), len(tasks), context)

class CuPCA:
    def __init__(self):
        self.ncclUniqueId = nccl.get_unique_id()

    def fit(self, df: DataFrame):
        sparkCtx = SparkSession.builder.getOrCreate()
        uniqueId = sparkCtx.sparkContext.broadcast(self.ncclUniqueId)
        topk = self.topK

        def partition_fit_functor(iterator):
            import time
            context_info = _context_info()

            nWorkers = context_info[2]
            wid = context_info[1]

            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(wid)
            ncclComm = nccl()
            ncclComm.init(nWorkers, uniqueId.value, wid)
            handle = Handle(n_streams = 0)
            
            if (wid == 1):
                time.sleep(3)
            print(uniqueId.value)
            print(nWorkers)
            print(wid)
            print("-----end inject_comms--" + str(wid))

            inject_comms_on_handle_coll_only(handle, ncclComm, nWorkers, wid, True)
            kwargs = {'n_components' : 1, 'whiten' : False}
            #cupyArraysList = [cp.array(list(iterator))]
            #pcaObject = CumlPCA(handle=handle, output_type = 'cupy', **kwargs)
            pcaObject = CumlPCA(output_type = 'cupy', **kwargs)
            #print(iterator)
            #print("---iterator----")
            time.sleep(5)
            yield 1


            #print(cupyArraysList)
            #print("----cupyArraysList----")

            #M, N = cupyArraysList[0].shape
            #partsToRanks = [[wid, M]]
            #rank = wid
            #_transform = False
            #pcaObject.fit(cupyArraysList, M, N, partsToRanks, rank, _transform)

            #print(type(cupyArraysList[0]))
            #print(cupyArraysList[0])
            #print("-----==----type(cupyArraysList[0])---=----")
            #res = pcaObject.transform(cupyArraysList[0])
            #print(type(res))
            #print(res)
            #print("-------end res----------")
            #yield pcaObject 

        barrierRDD = df.rdd.map(lambda row : row[self.inputCol]).barrier() 
        self.modelRDD = barrierRDD.mapPartitions(partition_fit_functor).cache()
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
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    numPart = 2
    topk = 1
    spark = SparkSession.builder.master("local[" + str(numPart) + "]").getOrCreate()
    #spark = SparkSession.builder \
    #        .master("spark://jinfengl-dt:7077") \
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