import joblib

import cloudpickle

import warnings

import numpy as np

import pandas as pd

import xgboost
from xgboost import XGBRegressor

import streamlit as st

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import(
    OneHotEncoder,
    MinMaxScaler,
    PowerTransformer,
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
    RareLabelEncoder,
    MeanEncoder,
    CountFrequencyEncoder
)

sklearn.set_config(transform_output="pandas") #to ensure sklearn transformers are returned as pandas dataframe not numpy array

# convenience functions


# preprocessing operations

# airline
air_transformer=Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1,replace_with="Others",n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

#doj
feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]


doj_transformer = Pipeline(steps=[
    ("dt",DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
    ("scaler", MinMaxScaler())
])

# source & destination
loc_transformer= Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

# dep_time & arrival_time
time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour","minute"])),
    ("scaler",MinMaxScaler())
])


def part_of_day(X):
    columns = X.columns.to_list()
    X_temp = X.assign(**{
        col : pd.to_datetime(X.loc[:,col]).dt.hour
        for col in columns
    })

    return(
        X_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:,col].between(4,12, inclusive="left"),
                 X_temp.loc[:,col].between(12,16, inclusive="left"),
                 X_temp.loc[:,col].between(16,20, inclusive="left")],
                ["morning","afternoon","evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

    
time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])


time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

# duration
def dur_cat(X):
    return(
        X
        .assign(duration_cat=np.select([X.duration.lt(180),
                                       X.duration.between(180,420,inclusive="left")],
                                       ["short","medium"],
                                       default="long"))
        .drop(columns="duration")
    )

    
def is_over(X):
	return (
		X
        .assign(duration_over_1000 = X.duration.ge(1000).astype(int))
		.drop(columns="duration")
	)

    
dur_pipe1 = Pipeline(steps=[
    ("cat",FunctionTransformer(func=dur_cat)),
    ("encoder",OrdinalEncoder(categories=[["short","medium","long"]]))
])

dur_union = FeatureUnion(transformer_list=[
    ("part1",dur_pipe1),
    ("part2",FunctionTransformer(func=is_over)),
    ("scaler",StandardScaler())
])

dur_transformer = Pipeline(steps=[
    ("outlier", Winsorizer(capping_method="iqr",fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union", dur_union)
])

# total_stops
def is_direct(X):
    return (
        X
        .assign(is_direct_flight= X.total_stops.eq(0).astype(int))
    )

total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("direct", FunctionTransformer(func= is_direct))
])

# additional_info
info_pipe1= Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="other")),
    ("encoder", OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
])

def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1),
    ("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("union", info_union)
])

# column transformer
column_transformer = ColumnTransformer(transformers=[
    ("air", air_transformer, ["airline"]),
    ("doj", doj_transformer, ["date_of_journey"]),
    ("loc", loc_transformer, ["source", "destination"]),
    ("time", time_transformer, ["dep_time","arrival_time"]),
    ("dur", dur_transformer, ["duration"]),
    ("stops", total_stops_transformer, ["total_stops"]),
    ("info", info_transformer, ["additional_info"])
], remainder="passthrough")

# feature selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
	estimator=estimator,
	scoring="r2",
	threshold=0.1
) 

#preprocessor
preprocessor = Pipeline(steps=[
	("ct", column_transformer),
	("selector", selector)
])

# reading training data
 
train = pd.read_csv("data/train.csv")
X_train = train.drop(columns="price")
y_train = train.price.copy()

# fit and save the preprocessor

preprocessor.fit(X_train,y_train)
with open("preprocessor.joblib", "wb") as f:
    cloudpickle.dump(preprocessor, f)

# WEB APPLICATION

st.set_page_config (
     page_title="Flights Prices Prediction",
     page_icon="✈️",
     layout="wide",
)

st.title("Flight Prices Prediction")

# user inputs
airline = st.selectbox(
     "Airline:",
     options = X_train.airline.unique()
)

doj = st.date_input("Date of Journey:")

source = st.selectbox(
     "Source:",
     options = X_train.source.unique()
)

destination = st.selectbox(
     "Destination:",
     options = X_train.destination.unique()
)

dep_time = st.time_input("Departure Time:")

arrival_time = st.time_input("Arrival Time:")

duration = st.number_input(
     "Duration (mins):",
     step=1
)

total_stops = st.number_input(
     "Total Stops:",
     step=1,
     min_value=0
)

additional_info = st.selectbox(
     "Additional info:",
     options = X_train.additional_info.unique()
)

x_new = pd.DataFrame(dict(
     airline=[airline],
     date_of_journey=[doj],
     source=[source],
     destination=[destination],
     dep_time=[dep_time],
     arrival_time=[arrival_time],
     duration=[duration],
     total_stops=[total_stops],
     additional_info=[additional_info]
)).astype({
     col: "str"
     for col in ["date_of_journey","dep_time","arrival_time"]
}) #pandas to_datetime doesn't work on streamlit date-time object

if st.button("Predict"):
     with open("preprocessor.joblib", "rb") as f:
        saved_preprocessor = cloudpickle.load(f)

     x_new_pre = saved_preprocessor.transform(x_new)
     
     model = joblib.load("rf-model.pkl")
     pred = model.predict(x_new_pre)[0]

     st.info(f"The Predicted Price is {pred:,.0f} INR")

