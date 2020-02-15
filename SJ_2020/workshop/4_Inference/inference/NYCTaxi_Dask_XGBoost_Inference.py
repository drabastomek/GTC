# Import modules
import dask
import dask_cudf
# import cudf
import os
import socket
import dask_xgboost as dxgb
import distributed
import pickle
import glob
from collections import OrderedDict
import math
from math import cos, sin, asin, sqrt, pi
import numpy as np
import datetime

from dask.delayed import delayed
from azureml.core import Run

if __name__ == '__main__':
    t_start = datetime.datetime.now()
    ### PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler")
    parser.add_argument("--datastore_name")
    parser.add_argument("--path_to_data")
    parser.add_argument("--path_to_model")
    parser.add_argument("--path_to_output", default='.')
    parser.add_argument("--file_index_start", default=0)
    parser.add_argument("--file_index_end", default=13)

    args, unparsed = parser.parse_known_args()
    print(args, unparsed)
    # Setup Dask cluster and communications

    print("Setting dask settings...")
    dask.config.set({'distributed.scheduler.work-stealing': False})
    dask.config.set({'distributed.scheduler.bandwidth': 20})
    print("Changes to dask settings")
    print("-> Setting work-stealing to ", dask.config.get('distributed.scheduler.work-stealing'))
    print("-> Setting scheduler bandwidth to ", dask.config.get('distributed.scheduler.bandwidth'))
    print("Settings updates complete")

    run = Run.get_context()
    ip = socket.gethostbyname(socket.gethostname())

    client = distributed.Client(args.scheduler)
    client.restart()

    ### SETUP storage options
    datastore = (
            run
            .experiment
            .workspace
            .datastores[args.datastore_name]
    )

    STORAGE_OPTIONS = {
        'account_name': datastore.account_name
        , 'account_key' : datastore.account_key
    }

    protocol = 'abfs'
    container = 'datasets' # default for Workspace

    ### DATAFILES
    path_split = args.path_to_data.split('/')
    idx = path_split.index(args.datastore_name)
    start, end = int(args.file_index_start), int(args.file_index_end)

    datafiles = glob.glob(args.path_to_data)
    datafiles = ['/'.join(f.split('/')[idx+1:]) for f in datafiles[start:end]]

    files_to_read = [
        f'{protocol}://{container}/{f}' 
        for f 
        in datafiles
    ]

    run.log_list('files_to_read', files_to_read) # log the filenames of files used
    
    ### data transformations
    remap = {}
    remap['tpep_pickup_datetime'] = 'pickup_datetime'
    remap['tpep_dropoff_datetime'] = 'dropoff_datetime'
    remap['ratecodeid'] = 'rate_code'

    #create a list of columns & dtypes the df must have
    must_haves = {
        'pickup_datetime': 'datetime64[ms]',
        'dropoff_datetime': 'datetime64[ms]',
        'passenger_count': 'int32',
        'trip_distance': 'float32',
        'pickup_longitude': 'float32',
        'pickup_latitude': 'float32',
        'rate_code': 'int32',
        'dropoff_longitude': 'float32',
        'dropoff_latitude': 'float32',
        'fare_amount': 'float32'
    }

    # helper function which takes a DataFrame partition
    def clean(df_part, remap, must_haves):    
        # some col-names include pre-pended spaces remove & lowercase column names
        tmp = {col:col.strip().lower() for col in list(df_part.columns)}
        df_part = df_part.rename(tmp)
        
        # rename using the supplied mapping
        df_part = df_part.rename(remap)
        
        # iterate through columns in this df partition
        for col in df_part.columns:
            # drop anything not in our expected list
            if col not in must_haves:
                df_part = df_part.drop(col)
                continue

            if df_part[col].dtype == 'object' and col in ['pickup_datetime', 'dropoff_datetime']:
                df_part[col] = df_part[col].astype('datetime64[ms]')
                continue
                
            # if column was read as a string, recast as float
            if df_part[col].dtype == 'object':
                df_part[col] = df_part[col].str.fillna('-1')
                df_part[col] = df_part[col].astype('float32')
            else:
                # downcast from 64bit to 32bit types
                if 'int' in str(df_part[col].dtype):
                    df_part[col] = df_part[col].astype('int32')
                if 'float' in str(df_part[col].dtype):
                    df_part[col] = df_part[col].astype('float32')
                df_part[col] = df_part[col].fillna(-1)
        
        return df_part

    taxi_df = (
        dask_cudf
        .read_csv(files_to_read, storage_options=STORAGE_OPTIONS)
            .repartition(npartitions=32)
            .map_partitions(clean, remap, must_haves)
            .persist()
    )
        
    number_of_records = len(taxi_df)
    run.log('number_of_records', number_of_records)

    # apply a list of filter conditions to throw out records with missing or outlier values
    query_frags = [
        'fare_amount > 0 and fare_amount < 500',
        'passenger_count > 0 and passenger_count < 6',
        'pickup_longitude > -75 and pickup_longitude < -73',
        'dropoff_longitude > -75 and dropoff_longitude < -73',
        'pickup_latitude > 40 and pickup_latitude < 42',
        'dropoff_latitude > 40 and dropoff_latitude < 42'
    ]
    taxi_df = taxi_df.query(' and '.join(query_frags))

    def haversine_distance_kernel(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, h_distance):
        for i, (x_1, y_1, x_2, y_2) in enumerate(zip(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)):
            x_1 = pi / 180 * x_1
            y_1 = pi / 180 * y_1
            x_2 = pi / 180 * x_2
            y_2 = pi / 180 * y_2
            
            dlon = y_2 - y_1
            dlat = x_2 - x_1
            a = sin(dlat / 2)**2 + cos(x_1) * cos(x_2) * sin(dlon / 2)**2
            
            c = 2 * asin(sqrt(a)) 
            r = 6371 # Radius of earth in kilometers
            
            h_distance[i] = c * r

    def day_of_the_week_kernel(day, month, year, day_of_week):
        for i, (d_1, m_1, y_1) in enumerate(zip(day, month, year)):
            if month[i] < 3:
                shift = month[i]
            else:
                shift = 0
            Y = year[i] - (month[i] < 3)
            y = Y - 2000
            c = 20
            d = day[i]
            m = month[i] + shift + 1
            day_of_week[i] = (d + math.floor(m * 2.6) + y + (y // 4) + (c // 4) - 2 * c) % 7
            
    def add_features(df):
        df['hour'] = df['pickup_datetime'].dt.hour
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['diff'] = df['dropoff_datetime'].astype('int32') - df['pickup_datetime'].astype('int32')

        df['pickup_latitude_r']   = (df['pickup_latitude']   / .01).astype('int') / 100.0
        df['pickup_longitude_r']  = (df['pickup_longitude']  / .01).astype('int') / 100.0
        df['dropoff_latitude_r']  = (df['dropoff_latitude']  / .01).astype('int') / 100.0
        df['dropoff_longitude_r'] = (df['dropoff_longitude'] / .01).astype('int') / 100.0

        
        df = df.drop('pickup_datetime')
        df = df.drop('dropoff_datetime')

        df = df.apply_rows(haversine_distance_kernel,
                        incols=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'],
                        outcols=dict(h_distance=np.float32),
                        kwargs=dict())

        df = df.apply_rows(day_of_the_week_kernel,
                        incols=['day', 'month', 'year'],
                        outcols=dict(day_of_week=np.float32),
                        kwargs=dict())


        df['is_weekend'] = (df['day_of_week']<2).astype("int32")
        return df
    
    # actually add the features
    taxi_df = taxi_df.map_partitions(add_features).persist()
    done = distributed.wait(taxi_df)

    filtered_number_of_rows = len(taxi_df)
    run.log('filtered_number_of_rows', filtered_number_of_rows)

    def drop_empty_partitions(df):
        lengths = df.map_partitions(len).compute()
        nonempty = [length > 0 for length in lengths]
        return df.partitions[nonempty]

    model = pickle.load(open(args.path_to_model, 'rb'))
    print(model)

    label = taxi_df[['fare_amount']].persist()
    features = taxi_df
    features = features.drop('fare_amount', axis=1)
    features = drop_empty_partitions(features)
    features = features[model.feature_names]
    done = distributed.wait([features, label])

    ### PREDICT
    label['predictions'] = dxgb.predict(client, model, features)
    label.to_csv(os.path.join(args.path_to_output, 'labels_pred.csv'))
    
    run.log('Elapsed time', str(datetime.datetime.now() - t_start))