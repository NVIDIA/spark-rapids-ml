from enum import unique
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import ArrayType 
from pyspark.sql.types import DoubleType
from pyspark import RDD
from pyspark.sql import DataFrame

import cupy as cp

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

        dimension = len(df.first()[self.inputCol])
        numVec = df.count()

        def part2rankFunc(iter):
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            pid = context.partitionId()
            size = sum([1 for x in iter])
            yield (pid, size)

        part2rank = df.rdd.barrier().mapPartitions(part2rankFunc).collect()
        print(part2rank)
        print("---part2rank--")

        def partition_fit_functor(iterator, numVec, dimension, partsToRanks):
            import time
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            tasks = context.getTaskInfos()
            context_info = (tasks[0].address.split(":")[0], context.partitionId(), len(tasks), context)

            nWorkers = context_info[2]
            wid = context_info[1]

            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(wid)
            ncclComm = nccl()

            rootUniqueId = uniqueId.value
            ncclComm.init(nWorkers, rootUniqueId, wid)
            handle = Handle(n_streams = 0)
            
            if (wid == 1):
                time.sleep(1)
            print(rootUniqueId)
            print(nWorkers)
            print(wid)
            print("......worker information from ---" + str(wid))

            inject_comms_on_handle_coll_only(handle, ncclComm, nWorkers, wid, True)
            kwargs = {'n_components' : 1, 'whiten' : False}
            cupyArraysList = [cp.array(list(iterator))]
            print(cupyArraysList)
            print("----cupyArraysList----" + str(wid))

            M = numVec
            N =  dimension
            rank = wid
            _transform = False

            from cuml.decomposition.pca_mg import PCAMG as CumlPCA
            pcaObject = CumlPCA(handle=handle, output_type = 'cupy', **kwargs)
            pcaObject.fit(cupyArraysList, M, N, partsToRanks, rank, _transform)

            res = pcaObject.transform(cupyArraysList[0])
            print(type(res))
            print(res)
            print("-------end res----------" + str(wid))
            yield pcaObject 

        barrierRDD = df.rdd.map(lambda row : row[self.inputCol]).barrier() 
        self.modelRDD = barrierRDD.mapPartitions(lambda iter : partition_fit_functor(iter, numVec, dimension, part2rank)).cache()
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
    numPart = 1
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