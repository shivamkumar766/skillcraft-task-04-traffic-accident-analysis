import pandas as pd

# Load Dataset
file_path = "dft-road-casualty-statistics-collision-2025.csv"

df = pd.read_csv(file_path, low_memory=False)

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display total rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display all column names
print("\nAll Column Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check dataset data types
print("\nData Types:")
print(df.dtypes)


# Step 6: Data Cleaning and Preparation

print("\n--- Data Cleaning Started ---")

# Check duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

# Remove duplicate records
df = df.drop_duplicates()

# Convert date column
df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

# Extract accident hour
df["accident_hour"] = pd.to_datetime(
    df["time"],
    format="%H:%M",
    errors="coerce"
).dt.hour

# Convert latitude and longitude
df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
)

# Display cleaned data
print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nData Cleaning Completed Successfully!")


# Step 7: Accident Pattern Analysis

print("\n========== ACCIDENT PATTERN ANALYSIS ==========")

# 1. Weather Conditions Analysis
print("\nAccidents by Weather Conditions:")
print(df["weather_conditions"].value_counts().sort_index())

# 2. Road Surface Conditions Analysis
print("\nAccidents by Road Surface Conditions:")
print(df["road_surface_conditions"].value_counts().sort_index())

# 3. Time of Day Analysis
print("\nAccidents by Hour:")
print(df["accident_hour"].value_counts().sort_index())

# 4. Accident Severity Analysis
print("\nAccidents by Severity:")
print(df["collision_severity"].value_counts().sort_index())

# 5. Road Type Analysis
print("\nAccidents by Road Type:")
print(df["road_type"].value_counts().sort_index())

print("\nAccident Pattern Analysis Completed!")


# Step 8: Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)

sns.set_style("whitegrid")

# 1. Weather Conditions Graph
plt.figure(figsize=(10, 6))
df["weather_conditions"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents by Weather Conditions")
plt.xlabel("Weather Condition Code")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/weather_conditions.png")
plt.show()

# 2. Road Surface Conditions Graph
plt.figure(figsize=(10, 6))
df["road_surface_conditions"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents by Road Surface Conditions")
plt.xlabel("Road Surface Condition Code")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/road_surface_conditions.png")
plt.show()

# 3. Accidents by Hour Graph
plt.figure(figsize=(10, 6))
df["accident_hour"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/accidents_by_hour.png")
plt.show()

# 4. Accident Severity Graph
plt.figure(figsize=(8, 5))
df["collision_severity"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents by Severity")
plt.xlabel("Severity Code")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/accident_severity.png")
plt.show()

print("\nAll Graphs Created and Saved Successfully!")


# Step 9: Accident Hotspot Map Visualization

import folium
from folium.plugins import HeatMap

print("\nCreating Accident Hotspot Map...")

# Select latitude and longitude
location_data = df[["latitude", "longitude"]].dropna()

# Keep valid Great Britain coordinates
location_data = location_data[
    (location_data["latitude"].between(49, 61)) &
    (location_data["longitude"].between(-8, 2))
]

# Sample records for faster visualization
if len(location_data) > 20000:
    location_data = location_data.sample(
        n=20000, random_state=42
    )

# Create UK map

accident_map = folium.Map(
    location=[54.5, -2.5],
    zoom_start=6,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri"
)


# Prepare heatmap data
heat_data = location_data[
    ["latitude", "longitude"]
].values.tolist()

# Add accident heatmap
HeatMap(
    heat_data,
    radius=10,
    blur=15,
    min_opacity=0.3
).add_to(accident_map)

# Save map
accident_map.save("graphs/accident_hotspot_map.html")

print("\nValid Accident Locations:", len(location_data))
print("Accident Hotspot Map Created Successfully!")


# Step 10: Advanced Accident Analysis

print("\n========== ADVANCED ACCIDENT ANALYSIS ==========")

# 1. Day-wise Accident Analysis
print("\nDay-wise Accident Count:")
print(df["day_of_week"].value_counts().sort_index())

plt.figure(figsize=(9, 5))
df["day_of_week"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents by Day of Week")
plt.xlabel("Day of Week Code")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/day_wise_accidents.png")
plt.show()

# 2. Speed Limit Analysis
print("\nAccidents by Speed Limit:")
print(df["speed_limit"].value_counts().sort_index())

plt.figure(figsize=(9, 5))
df["speed_limit"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents by Speed Limit")
plt.xlabel("Speed Limit (mph)")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/speed_limit_accidents.png")
plt.show()

# 3. Urban vs Rural Analysis
print("\nUrban and Rural Accident Count:")
print(df["urban_or_rural_area"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
df["urban_or_rural_area"].value_counts().sort_index().plot(kind="bar")
plt.title("Road Accidents: Urban vs Rural")
plt.xlabel("Area Type Code")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/urban_rural_accidents.png")
plt.show()

print("\nAdvanced Accident Analysis Completed Successfully!")


# Step 11: Convert Codes into Meaningful Labels

# Day of Week Mapping
day_mapping = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}

# Accident Severity Mapping
severity_mapping = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

# Urban and Rural Mapping
area_mapping = {
    1: "Urban",
    2: "Rural",
    3: "Unallocated"
}

# Create readable columns
df["day_name"] = df["day_of_week"].map(day_mapping)
df["severity_name"] = df["collision_severity"].map(severity_mapping)
df["area_name"] = df["urban_or_rural_area"].map(area_mapping)

# Display readable analysis
print("\nDay-wise Accidents:")
print(df["day_name"].value_counts())

print("\nAccident Severity:")
print(df["severity_name"].value_counts())

print("\nUrban vs Rural Accidents:")
print(df["area_name"].value_counts())

# Create labeled severity graph
plt.figure(figsize=(8, 5))
df["severity_name"].value_counts().plot(kind="bar")
plt.title("Accidents by Severity")
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/severity_labeled.png")
plt.show()

# Create labeled area graph
plt.figure(figsize=(8, 5))
df["area_name"].value_counts().plot(kind="bar")
plt.title("Urban vs Rural Accidents")
plt.xlabel("Area Type")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("graphs/area_labeled.png")
plt.show()

print("\nCode Mapping and Analysis Completed!")


# Step 12: Weather and Road Surface Code Mapping

# Weather Conditions Mapping
weather_mapping = {
    0: "Unknown / Not Recorded",
    1: "Fine without high winds",
    2: "Raining without high winds",
    3: "Snowing without high winds",
    4: "Fine with high winds",
    5: "Raining with high winds",
    6: "Snowing with high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown"
}

# Road Surface Conditions Mapping
surface_mapping = {
    0: "Unknown / Not Recorded",
    1: "Dry",
    2: "Wet / Damp",
    3: "Snow",
    4: "Frost / Ice",
    5: "Flood",
    9: "Unknown"
}

# Create readable columns
df["weather_name"] = df["weather_conditions"].map(
    weather_mapping
).fillna("Other / Unmapped")

df["surface_name"] = df["road_surface_conditions"].map(
    surface_mapping
).fillna("Other / Unmapped")

# Display Weather Analysis
print("\nWeather Conditions Analysis:")
print(df["weather_name"].value_counts())

# Display Road Surface Analysis
print("\nRoad Surface Conditions Analysis:")
print(df["surface_name"].value_counts())

# Weather Graph
plt.figure(figsize=(12, 6))
df["weather_name"].value_counts().plot(kind="bar")
plt.title("Road Accidents by Weather Conditions")
plt.xlabel("Weather Conditions")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("graphs/weather_labeled.png")
plt.show()

# Road Surface Graph
plt.figure(figsize=(10, 6))
df["surface_name"].value_counts().plot(kind="bar")
plt.title("Road Accidents by Road Surface Conditions")
plt.xlabel("Road Surface Conditions")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("graphs/road_surface_labeled.png")
plt.show()

print("\nWeather and Road Surface Analysis Completed!")


# Step 13: Final Data Verification

print("\n========== FINAL DATA VERIFICATION ==========")

# Verify actual severity codes
print("\nCollision Severity Codes:")
print(df["collision_severity"].value_counts().sort_index())

# Verify area codes
print("\nUrban and Rural Codes:")
print(df["urban_or_rural_area"].value_counts().sort_index())

# Verify day names directly from date
df["day_name"] = df["date"].dt.day_name()

print("\nAccidents by Actual Day:")
print(df["day_name"].value_counts())

# Total accidents
print("\nTotal Accidents:", len(df))

# Total casualties
print("Total Casualties:", df["number_of_casualties"].sum())

# Average vehicles involved
print("Average Vehicles per Collision:",
      round(df["number_of_vehicles"].mean(), 2))

print("\nFinal Verification Completed!")
