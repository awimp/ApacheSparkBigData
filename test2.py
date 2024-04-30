from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Step 1: Setting Up Apache Spark
conf = SparkConf().setAppName("BigDataAnalysis").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Step 2: Data Loading and Preprocessing (Replace 'data_path' with your dataset path)
data_path = "TWO_CENTURIES_OF_UM_RACES.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
df.show(5)

# Step 3: Exploratory Data Analysis (EDA)
# Summary statistics
print("Summary statistics:")
df.describe().show()

# Count missing values per column
print("Count of values per column:")
df.select([count(col(c)).alias(c) for c in df.columns]).show()

# Step 4: Implementing Big Data Processing Tasks
# RDD Operations
rdd_data = df.rdd  # Convert DataFrame to RDD for RDD operations
# Example RDD operation: Count records
total_records = rdd_data.count()
print(f"Total number of records: {total_records}")

# Spark SQL and DataFrames
# Example: Group by event name and count number of finishers
print("Total number of finishers by event name:")
df.groupBy("Event name").agg(count("Event number of finishers").alias("Total Finishers")).show()

# Step 5: Data Analysis and Insight Extraction
# Perform advanced analysis (e.g., pattern recognition, clustering)
# Example: Find top 5 most common event names
print("Top 5 most common event names:")
top_events = df.groupBy("Event name").count().orderBy(desc("count")).limit(5)
top_events.show()

# Extract meaningful insights and patterns from the data
# Example: Identify age categories with highest average speed
print("Average speed by age category:")
df.groupBy("Athlete age category").agg({"Athlete average speed": "avg"}).orderBy(desc("avg(Athlete average speed)")).show()

feature_columns = ['Athlete year of birth', 'Athlete average speed']

# Create a vector assembler to combine selected features into a single vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
data_for_clustering = assembler.transform(df.select(feature_columns))

# Step 4: Clustering (K-means)
# Train a K-means clustering model
kmeans = KMeans(k=3, seed=1)  # Specify number of clusters (k) and random seed
model = kmeans.fit(data_for_clustering)

# Get cluster centers
cluster_centers = model.clusterCenters()
print("Cluster Centers:")
for center in cluster_centers:
    print(center)

# Step 5: Assigning Clusters to Data Points
# Predict cluster labels for each data point
clustered_df = model.transform(data_for_clustering)
clustered_df.show(10)  # Display clustered data (with cluster labels)

# Stop Spark session
spark.stop()
