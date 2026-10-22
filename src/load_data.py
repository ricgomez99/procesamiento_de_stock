from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Instantiate a new Spark Session\
spark = SparkSession \
    .builder \
    .appName('load-csv') \
    .master('local[*]') \
    .getOrCreate()

# Read the CSV file
df = spark.read \
    .option('header', True) \
    .option('inferSchema', True) \
    .csv('data/AAPL.csv')

df.show()
df.printSchema()
column = F.col('Close')

df.select(column).show()