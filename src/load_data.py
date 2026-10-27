from pyspark.sql import DataFrame, SparkSession, Column
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType
from pyspark.sql.window import Window
import re

# Instantiate a new Spark Session\
spark = SparkSession \
    .builder \
    .appName('load-csv') \
    .master('local[*]') \
    .getOrCreate()

# Read the CSV file
# Use double qoutes to escape special characters
def load_stock_data(symbol: str):
    df = spark.read \
        .option('header', True) \
        .option('inferSchema', True) \
        .option('multiline', True) \
        .option('escape', '"') \
        .csv(f'data/{symbol}.csv')

    return df

# Clean Column Names
def clean_df_column_names(df: DataFrame):
    old_cols = df.columns
    new_cols = []

    for c in old_cols:
        new_col = re.sub(r'[\s\-]+', '_', c.strip().lower())
        new_col = re.sub(r'[^0-9a-z_]+', '', new_col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')

        new_cols.append(new_col)

    for old, new in zip(old_cols, new_cols):
        if old != new:
            df = df.withColumnRenamed(old, new)

    return df

# Clean data with NA tokes with proper null
def clean_data_with_null(df: DataFrame):
    na_tokens = ['', 'na', 'n/a', 'none', 'null', '-', '--', 'unknown']
    
    for column, data_type in df.dtypes:
        if data_type == 'string':
            # replace non-breaking spaces with regular spaces
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), '\xa0', ' ')
            )
            # trim leading and trailing spaces
            df = df.withColumn(
                column,
                F.trim(F.col(column))
            )
            # replace multiple spaces with a single space
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), r'\s+', ' ')
            )
            # replace NA tokens with null
            df = df.withColumn(
                column,
                F.when(
                    F.lower(F.col(column)).isin(na_tokens),
                    None
                ) \
                .otherwise(F.col(column))
            )

    return df

def standardize_double_format(df: DataFrame):
    types = df.dtypes

    for column, data_type in types:
        if data_type == 'double':
            # Replace full-width zero to normal zero
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), '０', '0')
            )
            
            # Remove non-numeric, non-comma, non-dot, non-dash chars
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), r'[^0-9,\.-]', '')
            ) 
            
            # Remove dots(.) used as thousand separator
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), r'\.', '')
            )
            
            # Replace comma with a dot for decimal conversion
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), r',', '.')
            )

            # Convert to double
            df = df.withColumn(
                column,
                F.col(column).cast('double')
            )

    return df

def standardize_date_format(df: DataFrame):
    df = df.withColumn('date', F.trim(F.col('date')))

    # Date separator: /
    df = df.withColumn(
        'date',
        F.regexp_replace(F.col('date'), r'\.', '-')
    )


    # Fill missing time (H:mm) only if date exists
    df = df.withColumn(
        'date',
        F.when(
            F.col('date').isNotNull() & (~F.col('date').rlike(r'\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{2}')),
            F.concat_ws(' ', F.col('date'), F.lit('00:00'))
        ).otherwise(F.col('date'))
    )


    # All dates should have the same format, Invalid formats should be NULL
    df = df.withColumn(
        'date_ts',
        F.try_to_timestamp(F.col('date'), F.lit('yyyy-MM-dd H:mm'))
    )

    # Format timestamp as YYYY-MM-DD HH:MM
    df = df.withColumn(
        'date',
        F.when(F.col('date_ts').isNotNull(), F.date_format(F.col('date_ts'), 'yyyy-MM-dd HH:mm')).otherwise(None)
    ).drop('date_ts')

    return df
        


aapl_df = load_stock_data('AAPL')
aapl_df = clean_df_column_names(aapl_df)
aapl_df = clean_data_with_null(aapl_df)
aapl_df = standardize_double_format(aapl_df)
aapl_df = standardize_date_format(aapl_df)

# Standardize Column Names
aapl_df.show(5, truncate=False)
aapl_df.printSchema()