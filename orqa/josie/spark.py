import pandas as pd

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

from orqa.josie.datalake import DataLakeHandler 


def get_spark_session(dlh:DataLakeHandler, **spark_config):
    conf = SparkConf().setAll(list(spark_config.items()))
    spark = SparkSession(sparkContext=SparkContext.getOrCreate(conf=conf))

    # adjusting logging level to error
    spark.sparkContext.setLogLevel("ERROR")

    init_rdd = spark.sparkContext.parallelize([(id_num, id_name) for id_num, id_name in dlh.mapping_id.items()])

    init_rdd = (
        init_rdd
        # using Polars here maybe is not the best, but in this case
        # even if (py)spark uses fork() there shouldn't be issues
        # With has_header=False the headers will be included in the data itself
        .map(lambda tabid_tabf: (tabid_tabf[0], pd.read_parquet(f'{dlh.datalake_location}/tables/{tabid_tabf[1]}.csv', infer_schema_length=0, has_header=True, encoding='latin1').rows()))
        .map(lambda tid_tab: (tid_tab[0], tid_tab[1], dlh.valid_columns[tid_tab[0]]))
    )

    return spark, init_rdd
