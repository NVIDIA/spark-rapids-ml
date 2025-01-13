#
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
#

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from pyspark import SparkContext
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, monotonically_increasing_id

from .utils import _get_spark_session, _is_local, get_logger

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap

P = TypeVar("P", bound="_CumlParams")


class HasEnableSparseDataOptim(Params):
    """
    This is a Params based class inherited from XGBOOST: https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/params.py.
    It holds the variable to store the boolean config for enabling sparse data optimization.
    """

    enable_sparse_data_optim = Param(
        Params._dummy(),
        "enable_sparse_data_optim",
        "This param activates sparse data optimization for VectorUDT features column. "
        "If the param is not included in an Estimator class, "
        "Spark rapids ml always converts VectorUDT features column into dense arrays when calling cuml backend. "
        "If included, Spark rapids ml will determine whether to create sparse arrays based on the param value: "
        "(1) If None, create dense arrays if the first VectorUDT of a dataframe is DenseVector. Create sparse arrays if it is SparseVector."
        "(2) If False, create dense arrays. This is favorable if the majority of vectors are DenseVector."
        "(3) If True, create sparse arrays. This is favorable if the majority of the VectorUDT vectors are SparseVector.",
        typeConverter=TypeConverters.toBoolean,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(enable_sparse_data_optim=None)


class HasFeaturesCols(Params):
    """
    Mixin for param featuresCols: features column names for multi-column input.
    """

    featuresCols = Param(
        Params._dummy(),  # type: ignore
        "featuresCols",
        "features column names for multi-column input.",
        TypeConverters.toListString,
    )

    def __init__(self) -> None:
        super(HasFeaturesCols, self).__init__()

    def getFeaturesCols(self) -> List[str]:
        """
        Gets the value of featuresCols or its default value.
        """
        return self.getOrDefault(self.featuresCols)


class HasIDCol(Params):
    """
    Mixin for param idCol: ID for each row of input dataset for row matching.
    """

    idCol = Param(
        Params._dummy(),  # type: ignore
        "idCol",
        "id column name.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasIDCol, self).__init__()

    def getIdCol(self) -> str:
        """
        Gets the value of `idCol`.
        """
        return self.getOrDefault("idCol")

    def _ensureIdCol(self, df: DataFrame) -> DataFrame:
        """
        Ensure an id column exists in the input dataframe. Add the column if not exists.
        """
        dedup = False
        if not self.isSet("idCol"):
            while self.getIdCol() in df.columns:
                self._set(**{"idCol": self.getIdCol() + "_dedup"})
                dedup = True

        id_col_name = self.getIdCol()
        df_withid = df.select(monotonically_increasing_id().alias(id_col_name), "*")
        df_withid = (
            df
            if self.isSet("idCol") and not dedup
            else df.select(monotonically_increasing_id().alias(id_col_name), "*")
        )
        return df_withid


class VerboseTypeConverters(TypeConverters):
    @staticmethod
    def _toIntOrBool(value: Any) -> Union[int, bool]:
        if isinstance(value, bool):
            return value

        if TypeConverters._is_integer(value):
            return int(value)

        raise TypeError("Could not convert %s to Union[int, bool]" % value)


class HasVerboseParam(Params):
    """
    Parameter to enable displaying verbose messages from cuml.
    Refer to the cuML documentation for details on verbosity levels.
    """

    verbose: "Param[Union[int, bool]]" = Param(
        Params._dummy(),
        "verbose",
        "cuml verbosity level (False, True or an integer between 0 and 6).",
        typeConverter=VerboseTypeConverters._toIntOrBool,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(verbose=False)


class _CumlClass(object):
    """
    Base class for all _CumlEstimator and _CumlModel implemenations.

    Defines helper methods for mapping Spark ML Params to cuML class parameters.
    """

    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        """
        Return a mapping of Spark ML Param names to cuML parameter names, which is used maintain
        associations from Spark params to cuML parameters.

        If the Spark Param has no equivalent cuML parameter, the cuML name can be set to:
        - empty string, if a defined Spark Param should just be silently ignored, or
        - None, if a defined Spark Param should raise an error.

        For algorithms without a Spark equivalent, the mapping can be left empty, with the exception
        of parameters for which we override the cuML default value with our own: these should include an identity mapping, e.g. {"param": "param"}.

        Note: standard Spark column Params, e.g. inputCol, featureCol, etc, should not be listed
        in this mapping, since they are handled differently.

        Example
        -------

        .. code-block::python

            # For KMeans
            return {
                "distanceMeasure": "",
                "k": "n_clusters",
                "initSteps": "",
                "maxIter": "max_iter",
                "seed": "random_state",
                "tol": "tol",
                "weightCol": None,
            }
        """
        return {}

    @classmethod
    def _param_value_mapping(
        cls,
    ) -> Dict[str, Callable[[Any], Union[None, str, float, int]]]:
        """
        Return a dictionary of cuML parameter names and a function mapping their Spark ML Param
        values to cuML values of either str, float, or int type.

        The mapped function should return None for any unmapped input values.

        If it is desired that a cuML value be accepted as a valid input, it must be explicitly mapped to
        itself in the function (see "squared_loss" and "eig" in example below).

        Example
        -------

        .. code-block:: python

            # For LinearRegression
            return {
                "loss": lambda x: {
                    "squaredError": "squared_loss",
                    "huber": None,
                    "squared_loss": "squared_loss",
                }.get(x, None),
                "solver": lambda x: {
                    "auto": "eig",
                    "normal": "eig",
                    "l-bfgs": None,
                    "eig": "eig",
                }.get(x, None),
            }

        """
        return {}

    @abstractmethod
    def _get_cuml_params_default(self) -> Dict[str, Any]:
        """Return a dictionary of parameter names and their default values.

        Note, please don't import cuml class and inspect the signatures to
        get the parameters, since it may break the rule that spark-rapids-ml should
        run on the driver side without rapids dependencies"""
        raise NotImplementedError()


class _CumlParams(_CumlClass, HasVerboseParam, Params):
    """
    Mix-in to handle common parameters for all Spark Rapids ML algorithms, along with utilties
    for synchronizing between Spark ML Params and cuML class parameters.
    """

    _cuml_params: Dict[str, Any] = {}
    _num_workers: Optional[int] = None
    _float32_inputs: bool = True

    @property
    def cuml_params(self) -> Dict[str, Any]:
        """
        Returns the dictionary of parameters intended for the underlying cuML class.
        """
        return self._cuml_params

    @property
    def num_workers(self) -> int:
        """
        Number of cuML workers, where each cuML worker corresponds to one Spark task
        running on one GPU.
        """

        inferred_workers = self._infer_num_workers()
        if self._num_workers is not None:
            # user sets the num_workers explicitly
            sc = _get_spark_session().sparkContext
            if _is_local(sc):
                default_parallelism = sc.defaultParallelism
                if default_parallelism < self._num_workers:
                    raise ValueError(
                        f"The num_workers ({self._num_workers}) should be less than "
                        f"or equal to spark default parallelism ({default_parallelism})"
                    )
                elif inferred_workers < self._num_workers:
                    raise ValueError(
                        f"The num_workers ({self._num_workers}) should be less than "
                        f"or equal to total GPUs ({inferred_workers})"
                    )
            elif inferred_workers < self._num_workers:
                get_logger(self.__class__).warning(
                    f"Spark cluster may not have enough executors. "
                    f"Found {inferred_workers} < {self._num_workers}"
                )
            return self._num_workers

        return inferred_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._num_workers = value

    def copy(self: P, extra: Optional["ParamMap"] = None) -> P:
        """
        Create a copy of the current instance, including its parameters and cuml_params.

        This function extends the default `copy()` method to ensure the `cuml_params` variable
        is also copied. The default `super().copy()` method only handles `_paramMap` and
        `_defaultParamMap`.

        Parameters
        -----------
        extra : Optional[ParamMap]
            A dictionary or ParamMap containing additional parameters to set in the copied instance.
            Note ParamMap = Dict[pyspark.ml.param.Param, Any].

        Returns
        --------
        P
            A new instance of the same type as the current object, with parameters and
            cuml_params copied.

        Raises
        -------
        TypeError
            If any key in the `extra` dictionary is not an instance of `pyspark.ml.param.Param`.
        """
        # override this function to update cuml_params if possible
        instance: P = super().copy(extra)
        cuml_params = instance.cuml_params.copy()

        instance._cuml_params = cuml_params
        if isinstance(extra, dict):
            for param, value in extra.items():
                if isinstance(param, Param):
                    instance._set_params(**{param.name: value})
                else:
                    raise TypeError(
                        "Expecting a valid instance of Param, but received: {}".format(
                            param
                        )
                    )

        return instance

    def _initialize_cuml_params(self) -> None:
        """
        Set the default values of cuML parameters to match their Spark equivalents.
        """
        # initialize cuml_params with defaults from cuML
        self._cuml_params = self._get_cuml_params_default()

        # update default values from Spark ML Param equivalents
        param_map = self._param_mapping()

        for spark_param in param_map.keys():
            if self.hasDefault(spark_param):
                self._set_cuml_param(spark_param, self.getOrDefault(spark_param))

    def _set_params(self: P, **kwargs: Any) -> P:
        """
        Set the kwargs as Spark ML Params and/or cuML parameters, while maintaining parameter
        and value mappings defined by the _CumlClass.
        """
        param_map = self._param_mapping()

        # raise error if setting both sides of a param mapping
        for spark_param, cuml_param in param_map.items():
            if (
                spark_param != cuml_param
                and spark_param in kwargs
                and cuml_param in kwargs
            ):
                raise ValueError(
                    f"'{cuml_param}' is an alias of '{spark_param}', set one or the other."
                )

        for k, v in kwargs.items():
            if k == "inputCol":
                if isinstance(v, str):
                    self._set(**{"inputCol": v})
                elif isinstance(v, List):
                    self._set(**{"inputCols": v})
            elif k == "featuresCol":
                if isinstance(v, str):
                    self._set(**{"featuresCol": v})
                elif isinstance(v, List):
                    self._set(**{"featuresCols": v})
            elif self.hasParam(k):
                # Param is declared as a Spark ML Param
                self._set(**{str(k): v})  # type: ignore
                self._set_cuml_param(k, v, silent=False)
            elif k in self.cuml_params:
                # cuml param
                self._cuml_params[k] = v
                for spark_param, cuml_param in param_map.items():
                    if k == cuml_param:
                        # also set matching Spark Param, if exists
                        # TODO: map cuml values back to Spark equivalents?
                        try:
                            self._set(**{str(spark_param): v})
                        except TypeError:
                            # Spark params have a converter, which may not work
                            # as expected. Eg, it can't convert float back to
                            # str param.
                            # TypeError: Invalid param value given for param "featureSubsetStrategy".
                            # Could not convert <class 'float'> to string type
                            pass

            elif k == "num_workers":
                # special case, since not a Spark or cuML param
                self._num_workers = v
            elif k == "float32_inputs":
                self._float32_inputs = v
            else:
                raise ValueError(f"Unsupported param '{k}'.")
        return self

    def clear(self, param: Param) -> None:
        """
        Reset a Spark ML Param to its default value, setting matching cuML parameter, if exists.
        """
        super().clear(param)
        param_map = self._param_mapping()
        if param.name in param_map:
            cuml_param = param_map[param.name]
            if cuml_param:
                self._cuml_params[cuml_param] = self.getOrDefault(param.name)

    def _copy_cuml_params(self: P, to: P) -> P:
        """
        Copy this instance's cuml_params values into another instance, only setting parameters
        which already exist in the other instance.  This is intended to mirror the behavior of
        :py:func:`Params._copyValues()`.

        Parameters
        ----------
        to : :py:class:`_CumlParams`
            Other instance to copy parameter values into.

        Returns
        -------
        :py:class:`_CumlParams`
            Other instance.
        """
        for k, v in self._cuml_params.items():
            if k in to._cuml_params:
                to._cuml_params[k] = v
        return to

    def _get_input_columns(self) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Get input column(s) from any of inputCol, inputCols, featuresCol, or featuresCols.

        Single-column setters, e.g. `setInputCol`, should allow either a single col name,
        or a list of col names (to transparently support multi-column inputs), while storing
        values in the appropriate underlying params, e.g. `inputCol` or `inputCols`.

        Returns
        -------
        Tuple[Optional[str], Optional[List[str]]]
            tuple of either a single column name or a list of multiple column names.

        Raises
        ------
        ValueError
            if none of the four supported input column params are set.
        """
        input_col = None
        input_cols = None

        # Note: order is significant if multiple params are set, e.g. defaults vs. overrides
        if self.hasParam("inputCols") and self.isDefined("inputCols"):
            input_cols = self.getOrDefault("inputCols")
        elif self.hasParam("inputCol") and self.isDefined("inputCol"):
            input_col = self.getOrDefault("inputCol")
        elif self.hasParam("featuresCols") and self.isDefined("featuresCols"):
            input_cols = self.getOrDefault("featuresCols")
        elif self.hasParam("featuresCol") and self.isDefined("featuresCol"):
            input_col = self.getOrDefault("featuresCol")
        else:
            raise ValueError("Please set inputCol(s) or featuresCol(s)")

        return input_col, input_cols

    def _infer_num_workers(self) -> int:
        """
        Try to infer the number of cuML workers (i.e. GPUs in cluster) from the Spark environment.
        """
        num_workers = 1
        try:
            spark = _get_spark_session()
            if spark:
                sc = spark.sparkContext
                if _is_local(sc):
                    # assume using all local GPUs for Spark local mode
                    # TODO suggest using more CPUs (e.g. local[*]) if number of GPUs > number of CPUs
                    import cupy

                    num_workers = cupy.cuda.runtime.getDeviceCount()
                else:
                    num_executors = int(
                        spark.conf.get("spark.executor.instances", "-1")  # type: ignore
                    )
                    if num_executors == -1:
                        jsc = spark.sparkContext._jsc.sc()
                        num_executors = len(jsc.statusTracker().getExecutorInfos()) - 1

                    gpus_per_executor = float(
                        spark.conf.get("spark.executor.resource.gpu.amount", "1")  # type: ignore
                    )

                    num_workers = max(int(num_executors * gpus_per_executor), 1)
        except Exception as e:
            # ignore any exceptions and just use default value
            print(e)

        return num_workers

    def _get_cuml_param(self, spark_param: str, silent: bool = True) -> Optional[str]:
        param_map = self._param_mapping()

        if spark_param in param_map:
            cuml_param = param_map[spark_param]
            if cuml_param is None:
                if not silent:
                    # if Spark Param is mapped to None, raise error
                    raise ValueError(
                        f"Spark Param '{spark_param}' is not supported by cuML."
                    )
            elif cuml_param == "":
                # if Spark Param is mapped to empty string, warn and continue
                if not silent:
                    print(f"WARNING: Spark Param '{spark_param}' is not used by cuML.")
                cuml_param = None

            return cuml_param
        elif spark_param in self.cuml_params:
            # cuML param that is declared as a Spark param (e.g., for algos w/out Spark equivalents)
            return spark_param
        else:
            return None

    def _set_cuml_param(
        self, spark_param: str, spark_value: Any, silent: bool = True
    ) -> None:
        """Set a cuml_params parameter for a given Spark Param and value.
        The Spark Param may be a cuML param that is declared as a Spark param (e.g., for algos w/out Spark equivalents),
        in which case the cuML param will be returned from _get_cuml_param.

        Parameters
        ----------
        spark_param : str
            Spark ML Param name.
        spark_value : Any
            Value associated with the Spark ML Param.
        silent: bool
            Don't warn or raise errors, default=True.

        Raises
        ------
        ValueError
            If the Spark Param is explictly not supported.
        """

        cuml_param = self._get_cuml_param(spark_param, silent)

        if cuml_param is not None:
            # if Spark Param is mapped to cuML parameter, set cuml_params
            try:
                self._set_cuml_value(cuml_param, spark_value)
            except ValueError:
                # create more informative message
                param_ref_str = (
                    cuml_param + " or " + spark_param
                    if cuml_param != spark_param
                    else spark_param
                )
                raise ValueError(f"{param_ref_str} given invalid value {spark_value}")

    def _get_cuml_mapping_value(self, k: str, v: Any) -> Any:
        value_map = self._param_value_mapping()
        if k not in value_map:
            # no value mapping required
            return v
        else:
            # value map exists
            mapped_v = value_map[k](v)
            if mapped_v is not None:
                return mapped_v
            else:
                raise ValueError(f"Value '{v}' for '{k}' param is unsupported")

    def _set_cuml_value(self, k: str, v: Any) -> None:
        """
        Set a cuml_params parameter with a (mapped) value.

        If the value originated from a Spark ML Param, and a value mapping exists, the parameter
        will be set to the mapped value.  Generally, this is only useful for string/enum types.

        Parameters
        ----------
        k : str
            Name of a cuml_param parameter.
        v : Any
            Value to assign to the cuml_param parameter, which may be mapped to another value.

        Raises
        ------
        ValueError
            If a value mapping exists, but the mapped value is None, this means that there is
            no equivalent value for the cuML side, so an exception is raised.
        """
        value_map = self._get_cuml_mapping_value(k, v)
        self._cuml_params[k] = value_map


class DictTypeConverters(TypeConverters):
    @staticmethod
    def _toDict(value: Any) -> Dict[str, Any]:
        """
        Convert a value to a Dict type for Param typeConverter, if possible.
        Used to support Dict types with the Spark ML Param API.
        """
        if isinstance(value, Dict):
            return {TypeConverters.toString(k): v for k, v in value.items()}
        raise TypeError("Could not convert %s to Dict[str, Any]" % value)
