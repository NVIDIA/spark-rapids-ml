#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import SparkSession

from .utils import _get_spark_session, _is_local, get_logger

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap

P = TypeVar("P", bound="_CumlParams")


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


class _CumlParams(_CumlClass, Params):
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
        # override this function to update cuml_params if possible
        instance: P = super().copy(extra)
        cuml_params = instance.cuml_params.copy()

        if isinstance(extra, dict):
            for param, value in extra.items():
                if isinstance(param, Param):
                    name = instance._get_cuml_param(param.name, silent=False)
                    if name is not None:
                        cuml_params[name] = instance._get_cuml_mapping_value(
                            name, value
                        )
                else:
                    raise TypeError(
                        "Expecting a valid instance of Param, but received: {}".format(
                            param
                        )
                    )
        instance._cuml_params = cuml_params
        return instance

    def initialize_cuml_params(self) -> None:
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

    def set_params(self: P, **kwargs: Any) -> P:
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
            if self.hasParam(k):
                # standard Spark ML Param
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
            spark = SparkSession.getActiveSession()
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
        else:
            return None

    def _set_cuml_param(
        self, spark_param: str, spark_value: Any, silent: bool = True
    ) -> None:
        """Set a cuml_params parameter for a given Spark Param and value.

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
            self._set_cuml_value(cuml_param, spark_value)

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
