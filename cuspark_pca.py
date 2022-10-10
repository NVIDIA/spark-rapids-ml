from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import ArrayType 
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrame

from raft.dask.common.nccl import nccl
from raft.dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft.common import Handle
import time

# copy from Erik:torchdist_on_spark
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

        def partition_fit_functor(pdf_iterator, inputCol, numVec, dimension, partsToRanks, topk):
            from pyspark import BarrierTaskContext
            context = BarrierTaskContext.get()
            context.barrier()

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
            
            inject_comms_on_handle_coll_only(handle, ncclComm, nWorkers, wid, True)
            kwargs = {'n_components' : topk, 'whiten' : False}

            import cudf 
            cudfList = []
            for pdf in pdf_iterator:
                flatten = pdf.apply(lambda x : x[inputCol], axis = 1, result_type='expand')
                gdf = cudf.from_pandas(flatten)
                cudfList.append(gdf)

            M = numVec
            N =  dimension
            rank = wid
            _transform = False

            from cuml.decomposition.pca_mg import PCAMG as CumlPCA
            pcaObject = CumlPCA(handle=handle, output_type = 'cudf', **kwargs)
            pcaObject.fit(cudfList, M, N, partsToRanks, rank, _transform)

            res = pcaObject.transform(cudfList[0])
            print(type(res))
            print(res)
            print("-------end res----------" + str(wid) + "\n")
            print(pcaObject.mean_)
            print("-------end pcaObject.mean_----------" + str(wid) + "\n")

            import pandas
            if rank != 0:
                yield pandas.DataFrame({ "mean" : [[]], "pc" : [[]]})
            else:
                cpuMean = pcaObject.mean_.to_arrow().to_pylist()
                cpuPcFlat = pcaObject.components_.to_numpy().flatten()
                yield pandas.DataFrame({ "mean" : [cpuMean], "pc": [cpuPcFlat]})
        
        outSchema = StructType([
            StructField("mean", ArrayType(DoubleType(), False), False),
            StructField("pc", ArrayType(DoubleType(), False), False)
        ])

        # citation: mapInPandas barrier code refers to pyspark xgboost core.py 
        model = df.mapInPandas(
            lambda pdf_iter: partition_fit_functor(pdf_iter, self.inputCol, numVec, dimension, part2rank, topk), 
            schema=outSchema
            ) .rdd.barrier().mapPartitions(lambda x : x).collect()[0]
        print(model)

        print("----------in fit function------")
        return model 

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

    # in application code, numPart must be smaller or equal to the total number of GPUs
    numPart = 2

    topk = 1
    spark = SparkSession.builder.getOrCreate()

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
