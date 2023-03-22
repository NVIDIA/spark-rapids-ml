from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import cupy
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import SparkSession

from spark_rapids_ml.utils import _get_default_params_from_func, _is_local

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
    def _cuml_cls(cls) -> List[type]:
        """
        Return a list of cuML python counterpart class names, which will be used to
        auto-generate Spark params.
        """
        raise NotImplementedError

    @classmethod
    def _param_excludes(cls) -> List[str]:
        """
        Return a list of cuML class parameters which should not be auto-populated into the
        Spark class.

        Example
        -------

        .. code-block::python

            return [
                "handle",
                "output_type",
            ]

        """
        return []

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
    def _param_value_mapping(cls) -> Dict[str, Dict[str, Union[str, None]]]:
        """
        Return a dictionary of cuML parameter names and their mapping of Spark ML Param string
        values to cuML string values.

        If the mapped value is None, then the Spark value is unsupported, and an error will be
        raised.

        Example
        -------

        .. code-block:: python

            # For LinearRegression
            return {
                "loss": {
                    "squaredError": "squared_loss",
                    "huber": None,
                },
                "solver": {
                    "auto": "eig",
                    "normal": "eig",
                    "l-bfgs": None,
                }
            }

        """
        return {}

    @classmethod
    def _get_cuml_params_default(cls) -> Dict[str, Any]:
        """
        Inspect the __init__ function of associated _cuml_cls() to return a dictionary of
        parameter names and their default values.
        """
        params = {}
        for cls_type in cls._cuml_cls():
            params.update(
                _get_default_params_from_func(cls_type, cls._param_excludes())
            )
        return params


class _CumlParams(_CumlClass, Params):
    """
    Mix-in to handle common parameters for all Spark Rapids ML algorithms, along with utilties
    for synchronizing between Spark ML Params and cuML class parameters.
    """

    _cuml_params: Dict[str, Any] = {}
    _num_workers: Optional[int] = None

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
        return (
            self._infer_num_workers()
            if self._num_workers is None
            else self._num_workers
        )

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._num_workers = value

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
                    num_workers = cupy.cuda.runtime.getDeviceCount()
                else:
                    num_executors = int(
                        spark.conf.get("spark.executor.instances", "-1")
                    )
                    if num_executors == -1:
                        jsc = spark.sparkContext._jsc.sc()
                        num_executors = len(jsc.statusTracker().getExecutorInfos()) - 1

                    gpus_per_executor = float(
                        spark.conf.get("spark.executor.resource.gpu.amount", "1")
                    )
                    gpus_per_task = float(
                        spark.conf.get("spark.task.resource.gpu.amount", "1")
                    )

                    if gpus_per_task != 1:
                        msg = (
                            "WARNING: cuML requires 1 GPU per task, "
                            "'spark.task.resource.gpu.amount' is currently set to {}"
                        )
                        print(msg.format(gpus_per_task))
                        gpus_per_task = 1

                    num_workers = max(
                        int(num_executors * gpus_per_executor / gpus_per_task), 1
                    )
        except Exception as e:
            # ignore any exceptions and just use default value
            print(e)

        return num_workers

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
            else:
                # if Spark Param is mapped to cuML parameter, set cuml_params
                self._set_cuml_value(cuml_param, spark_value)

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
        value_map = self._param_value_mapping()
        if k not in value_map:
            # no value mapping required
            self._cuml_params[k] = v
        else:
            # value map exists
            supported_values = set([x for x in value_map[k].values() if x is not None])
            if v in supported_values:
                # already a valid value
                self._cuml_params[k] = v
            else:
                # try to map to a valid value
                mapped_v = value_map[k].get(v, None)
                if mapped_v:
                    self._cuml_params[k] = mapped_v
                else:
                    raise ValueError(
                        f"Value '{v}' for '{k}' param is unsupported, expected: {supported_values}"
                    )
