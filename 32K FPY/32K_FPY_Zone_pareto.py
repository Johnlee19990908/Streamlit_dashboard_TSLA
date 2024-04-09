import streamlit as st
import numpy as np
import clickhouse_connect
import pandas as pd 
import matplotlib.pyplot as plt
import ast
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
import streamlit as st
from datetime import date, timedelta 
import time

st.set_page_config(layout="wide")

st.title("32K Coherix FPY and Zone Failure Pareto")


# Calculate dates for the past 7 days
today = date.today()  # Correct usage of datetime.date.today()
seven_days_ago = today - timedelta(days=7)

# Create two date inputs for start and end date with default values
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", 
    [seven_days_ago, today]
)

# Ensure both dates are selected
if start_date and end_date:
    client = clickhouse_connect.get_client(
            host='xxxxxxxxx',
            port=8123,
            username='xxxxxxxxxxxx',
            password='xxxxxxxxxxxxxx',
            query_limit=0
        )
    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Update progress to indicate pre-query operations
    progress_bar.progress(10)
    time.sleep(0.5)  # Simulated delay

# Status
    status_text.text('Progress')
        # Modify the query to include the selected date range
    query = f'''
        SELECT
            TRACE_TAG,
            SERIAL_NUMBER,
            PROCESS_RESULT,
            PROP_PART,
            TRACE_TAG,
            TRACE_UNIT,
            KAFKA_TIMESTAMP,
            PROCESS_NAME,
            pts.EQUIPMENT,
            TRACE_VALUES
        FROM
            cybertruck.process_time_series pts
        WHERE
            PROCESS_NAME = 'FREM-BT2-PAK1-32000-Coherix-PD'
            and EQUIPMENT = 'FREM-BT2-PAK1-32000-Coherix'
            and pts.SERIAL_NUMBER IN (
                SELECT DISTINCT 
                    pts.SERIAL_NUMBER 
                FROM
                    cybertruck.process_time_series pts
                WHERE
                    PROCESS_NAME = 'FREM-BT2-PAK1-32000-Coherix-PD'
                    and EQUIPMENT = 'FREM-BT2-PAK1-32000-Coherix'
                    and KAFKA_TIMESTAMP BETWEEN '{start_date}' AND '{end_date}'
            )
            AND TRACE_TAG IN ('Height', 'Width', 'Zone')
        '''
    df1 = client.query_df(query)

    # Filter the DataFrame for TRACE_TAG = 'Height'
    height_df = df1[df1['TRACE_TAG'] == 'Height']

    # Initialize an empty DataFrame to hold the expanded data
    expanded_data = pd.DataFrame()

    # Iterate over each row in the height_df
    for index, row in height_df.iterrows():
        # Convert the TRACE_VALUES list into a DataFrame
        temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Height"])

        # Add SERIAL_NUMBER and KAFKA_TIMESTAMP columns to this DataFrame
        temp_df['SERIAL_NUMBER'] = row['SERIAL_NUMBER']
        temp_df['KAFKA_TIMESTAMP'] = row['KAFKA_TIMESTAMP']

        # Append this temp_df to the expanded_data DataFrame

        expanded_data = pd.concat([expanded_data, temp_df], ignore_index=True)
        
    # Adding a 'SERIAL_NUMBER index' column that counts the index within each SERIAL_NUMBER group
    expanded_data['SERIAL_NUMBER_index'] = expanded_data.groupby('SERIAL_NUMBER').cumcount()


    # Filter the DataFrame for TRACE_TAG = 'Width'
    width_df = df1[df1['TRACE_TAG'] == 'Width']

    # Initialize an empty DataFrame to hold the expanded data for Width
    expanded_width_data = pd.DataFrame()

    # Iterate over each row in the width_df
    for index, row in width_df.iterrows():
        # Convert the TRACE_VALUES list into a DataFrame with a 'Width' column
        temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Width"])

        # Append this temp_df to the expanded_width_data DataFrame
        expanded_width_data = pd.concat([expanded_width_data, temp_df], ignore_index=True)


    # Filter the DataFrame for TRACE_TAG = 'Zone'
    zone_df = df1[df1['TRACE_TAG'] == 'Zone']

    # Initialize an empty DataFrame to hold the expanded data for Zone
    expanded_zone_data = pd.DataFrame()

    # Iterate over each row in the zone_df
    for index, row in zone_df.iterrows():
        # Convert the TRACE_VALUES list into a DataFrame with a 'Zone' column
        temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Zone"])


        # Append this temp_df to the expanded_zone_data DataFrame
        expanded_zone_data = pd.concat([expanded_zone_data, temp_df], ignore_index=True)

    # Define the height limits for each zone
    height_limits = {
        1: {'max': 10, 'min': 4.5},
        2: {'max': 10.5, 'min': 4},
        3: {'max': 10, 'min': 4},
        4: {'max': 10, 'min': 5},
        5: {'max': 9, 'min': 4},
        6: {'max': 9, 'min': 4},
        7: {'max': 9, 'min': 4},
        8: {'max': 9, 'min': 4},
        9: {'max': 9, 'min': 4},
        10: {'max': 10, 'min': 4},
    }

    # Define the width limits for each zone
    width_limits = {
        1: {'max': 12, 'min': 5},
        2: {'max': 15.5, 'min': 8},
        3: {'max': 25, 'min': 7},
        4: {'max': 11.3, 'min': 6},
        5: {'max': 21, 'min': 3.5},
        6: {'max': 12, 'min': 7},
        7: {'max': 16, 'min': 7},
        8: {'max': 14, 'min': 8.5},
        9: {'max': 21, 'min': 3.5},
        10: {'max': 10.5, 'min': 7},
        # Additional zones can be added as needed
    }

    # Add the min and max limit columns for Height
    expanded_zone_data['Height min limit'] = expanded_zone_data['Zone'].map(lambda z: height_limits.get(z, {}).get('min', None))
    expanded_zone_data['Height max limit'] = expanded_zone_data['Zone'].map(lambda z: height_limits.get(z, {}).get('max', None))

    # Add the min and max limit columns for Width
    expanded_zone_data['Width min limit'] = expanded_zone_data['Zone'].map(lambda z: width_limits.get(z, {}).get('min', None))
    expanded_zone_data['Width max limit'] = expanded_zone_data['Zone'].map(lambda z: width_limits.get(z, {}).get('max', None))

    # Concatenate the three DataFrames along the columns
    merged_df = pd.concat([expanded_data, expanded_width_data, expanded_zone_data], axis=1)

    # Calculate the maximum index for each SERIAL_NUMBER in merged_df
    max_indices = merged_df.groupby('SERIAL_NUMBER')['SERIAL_NUMBER_index'].max()

    # Find SERIAL_NUMBERs with max index > 13000
    serial_numbers_to_remove = max_indices[max_indices > 1500000].index

    # Filter out rows in merged_df that have these SERIAL_NUMBERs
    filtered_merged_df = merged_df[~merged_df['SERIAL_NUMBER'].isin(serial_numbers_to_remove)]

    # Convert KAFKA_TIMESTAMP to datetime if it's not already
    filtered_merged_df['KAFKA_TIMESTAMP'] = pd.to_datetime(filtered_merged_df['KAFKA_TIMESTAMP'])


    # Sort the DataFrame by 'KAFKA_TIMESTAMP' in descending order
    filtered_merged_df = filtered_merged_df.sort_values(by='KAFKA_TIMESTAMP', ascending=False)

    # Extract the date part and create a new column 'Date'
    filtered_merged_df['Date'] = filtered_merged_df['KAFKA_TIMESTAMP'].dt.date

    filtered_merged_df['Date'] = pd.to_datetime(filtered_merged_df['Date'])

    filtered_merged_df = filtered_merged_df[~((filtered_merged_df['Height'] == 0.0) & (filtered_merged_df['Width'] == 0.0))]

    # Assuming 'filtered_merged_df' is your existing dataframe
    df = filtered_merged_df

    # Function to compute the test for height and width
    def compute_test(column_name, limits_dict, row):
        zone = row['Zone']
        value = row[column_name]
        
        if zone in limits_dict:
            if limits_dict[zone]['min'] <= value <= limits_dict[zone]['max']:
                return 0
            else:
                return 1
        else:
            return 0  # if the zone isn't found, return 0
############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(15)

    # Applying the function to create new columns
    df['Height_test'] = df.apply(lambda row: compute_test('Height', height_limits, row), axis=1)
    df['Width_test'] = df.apply(lambda row: compute_test('Width', width_limits, row), axis=1)
    df['either'] = df.apply(lambda row: 1 if row['Height_test'] == 1 or row['Width_test'] == 1 else 0, axis=1)

    # Fill NaN values with 0 and convert to integer data type
    df['Height_test'] = df['Height_test'].fillna(0).astype(int)
    df['Width_test'] = df['Width_test'].fillna(0).astype(int)

    # Create a new column 'Height_Width_test' that is the sum of 'Height_test' and 'Width_test'
    df = df.sort_values(by=["SERIAL_NUMBER", "SERIAL_NUMBER_index"])
    # Reset the SERIAL_NUMBER_index for each SERIAL_NUMBER group to start from 0
    df['SERIAL_NUMBER_index'] = df.groupby('SERIAL_NUMBER').cumcount()

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the cutoff date (31 days before the current date)
    cutoff_date = datetime.now() - timedelta(days=31)

    # Drop rows where 'Date' is earlier than 31 days
    #df = df[df['Date'] >= cutoff_date]


    
    #RTV Data preprocessing end here 


    ########################2K
    client = clickhouse_connect.get_client(host='xxxxxxxxxx', 

                            port=8123, username='xxxxxxxxx', 

                            password='xxxxxxxxxxxxx',

                            query_limit=0)
    query = f'''
        SELECT
            TRACE_TAG,
            SERIAL_NUMBER,
            PROCESS_RESULT,
            PROP_PART,
            TRACE_TAG,
            TRACE_UNIT,
            KAFKA_TIMESTAMP,
            PROCESS_NAME,
            pts.EQUIPMENT,
            TRACE_VALUES
        FROM
            cybertruck.process_time_series pts
        WHERE
            PROCESS_NAME = 'FREM-BT2-PAK1-32000-Coherix-LD'
            and EQUIPMENT = 'FREM-BT2-PAK1-32000-Coherix'
            and pts.SERIAL_NUMBER IN (
                SELECT DISTINCT 
                    pts.SERIAL_NUMBER 
                FROM
                    cybertruck.process_time_series pts
                WHERE
                    PROCESS_NAME = 'FREM-BT2-PAK1-32000-Coherix-LD'
                    and EQUIPMENT = 'FREM-BT2-PAK1-32000-Coherix'
                    and KAFKA_TIMESTAMP BETWEEN '{start_date}' AND '{end_date}'
            )
            AND TRACE_TAG IN ('Height', 'Width', 'Zone')
    '''
    df2 = client.query_df(query)
    # Filter the DataFrame for TRACE_TAG = 'Height'
    height_df = df2[df2['TRACE_TAG'] == 'Height']

    # Initialize an empty DataFrame to hold the expanded data
    expanded_data = pd.DataFrame()

    # Iterate over each row in the height_df
    for index, row in height_df.iterrows():
        # Convert the TRACE_VALUES list into a DataFrame
        temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Height"])

        # Add SERIAL_NUMBER and KAFKA_TIMESTAMP columns to this DataFrame
        temp_df['SERIAL_NUMBER'] = row['SERIAL_NUMBER']
        temp_df['KAFKA_TIMESTAMP'] = row['KAFKA_TIMESTAMP']

        # Append this temp_df to the expanded_data DataFrame

        expanded_data = pd.concat([expanded_data, temp_df], ignore_index=True)
        
    # Adding a 'SERIAL_NUMBER index' column that counts the index within each SERIAL_NUMBER group
    expanded_data['SERIAL_NUMBER_index'] = expanded_data.groupby('SERIAL_NUMBER').cumcount()


    # Filter the DataFrame for TRACE_TAG = 'Width'
    width_df = df2[df2['TRACE_TAG'] == 'Width']

############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(25)

    # Initialize an empty DataFrame to hold the expanded data for Width
    expanded_width_data = pd.DataFrame()

    # Iterate over each row in the width_df
    for index, row in width_df.iterrows():
        # Convert the TRACE_VALUES list into a DataFrame with a 'Width' column
        temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Width"])

        # Append this temp_df to the expanded_width_data DataFrame
        expanded_width_data = pd.concat([expanded_width_data, temp_df], ignore_index=True)


    # Filter the DataFrame for TRACE_TAG = 'Zone'
    zone_df = df2[df2['TRACE_TAG'] == 'Zone']

    # Initialize an empty DataFrame to hold the expanded data for Zone
    expanded_zone_data = pd.DataFrame()

    # Iterate over each row in the zone_df
    for index, row in zone_df.iterrows():
        # Convert the TRACE_VALUES list into a DataFrame with a 'Zone' column
        temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Zone"])


        # Append this temp_df to the expanded_zone_data DataFrame
        expanded_zone_data = pd.concat([expanded_zone_data, temp_df], ignore_index=True)

    # Define the height limits for each zone
    height_limits = {
        1: {'max': 10.5, 'min': 3.5},
        2: {'max': 10.5, 'min': 3},
        3: {'max': 10.5, 'min': 3},
        4: {'max': 10, 'min': 2.75},
        5: {'max': 10, 'min': 2.75},
    }

    # Define the width limits for each zone
    width_limits = {
        1: {'max': 9, 'min': 5},
        2: {'max': 10, 'min': 4},
        3: {'max': 10, 'min': 4},
        4: {'max': 10, 'min': 4},
        5: {'max': 10, 'min': 4},
    }


    # Add the min and max limit columns for Height
    expanded_zone_data['Height min limit'] = expanded_zone_data['Zone'].map(lambda z: height_limits.get(z, {}).get('min', None))
    expanded_zone_data['Height max limit'] = expanded_zone_data['Zone'].map(lambda z: height_limits.get(z, {}).get('max', None))

    # Add the min and max limit columns for Width
    expanded_zone_data['Width min limit'] = expanded_zone_data['Zone'].map(lambda z: width_limits.get(z, {}).get('min', None))
    expanded_zone_data['Width max limit'] = expanded_zone_data['Zone'].map(lambda z: width_limits.get(z, {}).get('max', None))

    # Concatenate the three DataFrames along the columns
    merged_df = pd.concat([expanded_data, expanded_width_data, expanded_zone_data], axis=1)

    # Calculate the maximum index for each SERIAL_NUMBER in merged_df
    max_indices = merged_df.groupby('SERIAL_NUMBER')['SERIAL_NUMBER_index'].max()

    # Find SERIAL_NUMBERs with max index > 13000
    serial_numbers_to_remove = max_indices[max_indices > 1300000].index

    # Filter out rows in merged_df that have these SERIAL_NUMBERs
    filtered_merged_df = merged_df[~merged_df['SERIAL_NUMBER'].isin(serial_numbers_to_remove)]

    # Convert KAFKA_TIMESTAMP to datetime if it's not already
    filtered_merged_df['KAFKA_TIMESTAMP'] = pd.to_datetime(filtered_merged_df['KAFKA_TIMESTAMP'])

    # Sort the DataFrame by 'KAFKA_TIMESTAMP' in descending order
    filtered_merged_df = filtered_merged_df.sort_values(by='KAFKA_TIMESTAMP', ascending=False)

    # Extract the date part and create a new column 'Date'
    filtered_merged_df['Date'] = filtered_merged_df['KAFKA_TIMESTAMP'].dt.date

    filtered_merged_df['Date'] = pd.to_datetime(filtered_merged_df['Date'])

    filtered_merged_df = filtered_merged_df[~((filtered_merged_df['Height'] == 0.0) & (filtered_merged_df['Width'] == 0.0))]

    # Assuming 'filtered_merged_df' is your existing dataframe
    df2 = filtered_merged_df
############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(30)
    # Function to compute the test for height and width
    def compute_test(column_name, limits_dict, row):
        zone = row['Zone']
        value = row[column_name]
        
        if zone in limits_dict:
            if limits_dict[zone]['min'] <= value <= limits_dict[zone]['max']:
                return 0
            else:
                return 1
        else:
            return 0  # if the zone isn't found, return 0

    # Applying the function to create new columns
    df2['Height_test'] = df2.apply(lambda row: compute_test('Height', height_limits, row), axis=1)
    df2['Width_test'] = df2.apply(lambda row: compute_test('Width', width_limits, row), axis=1)
    df2['either'] = df2.apply(lambda row: 1 if row['Height_test'] == 1 or row['Width_test'] == 1 else 0, axis=1)

    # Fill NaN values with 0 and convert to integer data type
    df2['Height_test'] = df2['Height_test'].fillna(0).astype(int)
    df2['Width_test'] = df2['Width_test'].fillna(0).astype(int)
############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(35)

    df2 = df2.sort_values(by=["SERIAL_NUMBER", "SERIAL_NUMBER_index"])

    # Reset the SERIAL_NUMBER_index for each SERIAL_NUMBER group to start from 0
    df2['SERIAL_NUMBER_index'] = df2.groupby('SERIAL_NUMBER').cumcount()

    # Convert 'Date' to datetime
    df2['Date'] = pd.to_datetime(df2['Date'])

    # Calculate the cutoff date (31 days before the current date)
    cutoff_date = datetime.now() - timedelta(days=31)

    # Drop rows where 'Date' is earlier than 31 days
    #df2 = df2[df2['Date'] >= cutoff_date]


    df2 = df2[~(df2[['Height', 'Width']] == 0.0).all(axis=1)]

    # Filter out rows where 'Zone' is equal to 0.0
    df2 = df2[df2['Zone'] != 0.0]

#2K Data preprocessing end here



    ###################################################### Create a Date Range Selector

    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    # Assuming max_date is defined as the latest date in your dataset
    max_date = filtered_merged_df['Date'].max()
   
    # Convert start_date and end_date to Pandas Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

        # 2. Filter the DataFrame Based on Selected Dates
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    ######################RTV failure
    counter = 0
    consecutive = []
    prev_serial = None
    
    for index, row in df.iterrows():  
            current_serial = row['SERIAL_NUMBER']
            if prev_serial and prev_serial != current_serial:
                counter = 0
    
            if row['either'] == 1:
                counter += 1
            else:
                counter = 0
    
            consecutive.append(counter)
            prev_serial = current_serial
    
    df['consecutive'] = consecutive  
    filtered_df_14 = df[df['consecutive'] == 14]  
    ############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(40)
    

    ###################### RTV FPY
    
    # Assuming the date column in your DataFrames is named 'Date'
    date_column = 'Date'
    
        # Calculate the total count per day
    total_count_per_day = df.groupby(date_column)['SERIAL_NUMBER'].nunique().reset_index(name='total_count')
    
        # Calculate the distinct failure count per day in filtered_df_14
    failure_count_per_day = filtered_df_14.groupby(date_column)['SERIAL_NUMBER'].nunique().reset_index(name='failure_count')
    
        # Merge the two dataframes
    merged_df = pd.merge(total_count_per_day, failure_count_per_day, on=date_column, how='left').fillna(0)
    
        # Convert the failure_count column to integer
    merged_df['failure_count'] = merged_df['failure_count'].astype(int)
    
        # Calculate the Daily First Pass Yield (FPY) and convert it to percentage
    merged_df['FPY'] = ((merged_df['total_count'] - merged_df['failure_count']) / merged_df['total_count']) * 100
    
        # Visualization
        # Create a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
        # Add total_count as a bar chart
    fig.add_trace(
            go.Bar(x=merged_df['Date'], y=merged_df['total_count'], name='Total Count', marker_color='blue'),
            secondary_y=False,
        )
    
        # Add failure_count as a bar chart, overlaid on total_count
    fig.add_trace(
            go.Bar(x=merged_df['Date'], y=merged_df['failure_count'], name='Failure Count', marker_color='red'),
            secondary_y=False,
        )
    
        # Update the layout
    fig.update_layout(
            title_text="RTV First Pass Yield (FPY) and Counts",
            xaxis_title="Date",
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=0.01
            ),
            barmode='overlay'
        )
    
        # Add FPY as a line chart on the secondary y-axis
    fig.add_trace(
            go.Scatter(
                x=merged_df['Date'],
                y=merged_df['FPY'],
                name='First Pass Yield (FPY)',
                mode='lines+markers',
                line=dict(color='green')
            ),
            secondary_y=True,
        )
    # Update progress to indicate pre-query operations
    progress_bar.progress(50)
    
        # Set y-axes titles
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="First Pass Yield (FPY) (%)", secondary_y=True)
    
        # Streamlit app layout
    st.title("RTV")
    st.plotly_chart(fig, use_container_width=True)
    
    
    ########################RTV heatmap
    ##############Date Range filter add here or Serial Number filter
    ############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(45)
    
        # Grouping by 'Zone' and listing all 'SERIAL_NUMBER' in that group, along with the count of occurrences
    grouped_df_zone = filtered_df_14.groupby('Zone').agg({'SERIAL_NUMBER': list, 'Height': 'count'}).reset_index()
    grouped_df_zone = grouped_df_zone.rename(columns={'Height': 'Count'})
    
        # Sorting the dataframe by 'Count' in descending order
    sorted_df = grouped_df_zone .sort_values(by='Count', ascending=False)
    
        # Use the rename() method to change the column name
    sorted_df.rename(columns={'Count': 'Failing_times'}, inplace=True)
    
        # Convert the "Zone" column to an integer type
    sorted_df['Zone'] = sorted_df['Zone'].astype(int)
        # Coordinates data
    coordinates_df = pd.DataFrame({
            'shapeId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'pointX': [1033, 513, 64, 46, 80, 187, 350, 633, 991, 1035],
            'pointY': [143, 69, 88, 319, 569, 579, 576, 577, 565, 440]
        })
    
    
    
        # Convert 'Failing_times' to integer if it's not already
    sorted_df['Failing_times'] = sorted_df['Failing_times'].astype(int)
    
        # Merge the dataframes
    merged_df = pd.merge(sorted_df, coordinates_df, left_on='Zone', right_on='shapeId')
    
    
        # Path to the uploaded image file
    image_path = r"C:\Users\hollee\Desktop\Denver\32K\raw data dashboard\32K_RTV.jpg"
        # Read the image file and encode it in base64
    with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        
    # Create a bubble plot with red bubbles and display 'Failing_times' on hover
    fig = px.scatter(
            merged_df,
            x='pointX',
            y='pointY',
            size='Failing_times',
            color_discrete_sequence=['red'], # Set bubbles to red
            hover_name='Failing_times', # Show 'Failing_times' on hover
            hover_data={'pointX': False, 'pointY': False} # Hide 'pointX' and 'pointY' from the hover data
        )
    
    
        # Adjust the text position
    fig.update_traces(textposition='top center')
    
        # Get the dimensions of the image
    img_width, img_height = 1079, 720
    
        # Use the encoded image string as source
    fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{encoded_string}',
                xref="x",
                yref="y",
                x=0,
                y=img_height,  # Assuming the origin (0,0) is at the bottom left
                sizex=img_width,
                sizey=img_height,
                sizing="contain",
                opacity=1.0,
                layer="below"
            )
        )
    
        # Update axes ranges to fit image
    fig.update_xaxes(range=[0, img_width])
    fig.update_yaxes(range=[0, img_height])
    
        # Update the layout to remove axes and gridlines
    fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            width=img_width,
            height=img_height,
            margin=dict(l=0, r=0, t=0, b=0)  # Remove margins to fit the image precisely
        )
    
        # Set the figure size to match the image size
    fig.update_layout(
            width=img_width,
            height=img_height,
            margin=dict(l=0, r=0, t=0, b=0)  # Remove margins to fit the image precisely
        )
    
    
    st.plotly_chart(fig, use_container_width=False)
    
    

############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(60)
        ###################### RTV Zone failure
    
        # Grouping by 'Zone' and listing all 'SERIAL_NUMBER' in that group, along with the count of occurrences
    grouped_df_zone = filtered_df_14.groupby('Zone').agg({'SERIAL_NUMBER': list, 'Height': 'count'}).reset_index()
    grouped_df_zone = grouped_df_zone.rename(columns={'Height': 'Count'})
    
        # Sorting the dataframe by 'Count' in descending order
    sorted_df = grouped_df_zone .sort_values(by='Count', ascending=False)
    
        # Use the rename() method to change the column name
    sorted_df.rename(columns={'Count': 'Failing_times'}, inplace=True)
    
        # Convert the "Zone" column to an integer type
    sorted_df['Zone'] = sorted_df['Zone'].astype(int)
    
        # Convert the integer "Zone" values to string without the decimal
    sorted_df['Zone'] = sorted_df['Zone'].astype(str)
    
        # Convert 'Zone' to string to maintain order
    sorted_df['Zone'] = sorted_df['Zone'].astype(str)
    
        # Get unique SERIAL_NUMBERS
    serial_numbers = sorted_df['SERIAL_NUMBER'].explode().unique()
    
    # Create a color scale for SERIAL_NUMBERS
    colors = px.colors.qualitative.Set1
    color_dict = {serial: colors[i % len(colors)] for i, serial in enumerate(serial_numbers)}

    # Initialize a Plotly figure
    fig = go.Figure()

    # Add bars for each SERIAL_NUMBER
    for serial in serial_numbers:
        serial_counts = sorted_df.apply(lambda x: x['SERIAL_NUMBER'].count(serial), axis=1)
        fig.add_trace(go.Bar(
            x=sorted_df['Zone'],
            y=serial_counts,
            name=str(serial),
            marker_color=color_dict[serial]
        ))

    # Calculate the cumulative percentage for the Pareto line
    sorted_df['Cumulative_Percentage'] = sorted_df['Failing_times'].cumsum() / sorted_df['Failing_times'].sum() * 100

    # Add the Pareto line with white color
    fig.add_trace(go.Scatter(
        x=sorted_df['Zone'],
        y=sorted_df['Cumulative_Percentage'],
        name='Cumulative Percentage',
        mode='lines+markers',
        marker=dict(color='white'),  # Set marker color to white
        line=dict(color='white'),    # Set line color to white
        yaxis='y2'
    ))


    # Update the layout for a stacked bar chart with a secondary y-axis for the Pareto line
    fig.update_layout(
    barmode='stack',
    title_text="RTV Zone Failure Pareto Analysis",
    xaxis_title='Zone',
    yaxis_title='Failing Times',
    yaxis2=dict(
        title='Cumulative Percentage',
        overlaying='y',
        side='right',
        showgrid=False,
        range=[0, 110]
    ),
    legend_title='SERIAL_NUMBER',
    xaxis={'type': 'category'},
    legend=dict(
        x=1.05,  # Adjust the x position of the legend
        y=1,
        orientation="v"
    )
    )

    st.plotly_chart(fig, use_container_width=True)
############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(65)

        
    ##################################################date range filter
    df2['Date'] = pd.to_datetime(df2['Date'])

    df2 = df2[
        (df2['Date'] >= start_date) &
        (df2['Date'] <= end_date)
    ]


    ######################2K failure
    counter = 0
    consecutive = []
    prev_serial = None

    for index, row in df2.iterrows():  # Change df to df2
        current_serial = row['SERIAL_NUMBER']
        if prev_serial and prev_serial != current_serial:
            counter = 0

        if row['either'] == 1:
            counter += 1
        else:
            counter = 0

        consecutive.append(counter)
        prev_serial = current_serial

    df2['consecutive'] = consecutive  # Change df to df2

    filtered_df_10 = df2[df2['consecutive'] == 10]  

# Update progress to indicate pre-query operations
    progress_bar.progress(75)

    ###################### 2K FPY

    date_column = 'Date'
    # Calculate the total count per day for df2
    total_count_per_day_df2 = df2.groupby(date_column)['SERIAL_NUMBER'].nunique().reset_index(name='total_count')

    # Calculate the distinct failure count per day in filtered_df2_14
    failure_count_per_day_df2 = filtered_df_10.groupby(date_column)['SERIAL_NUMBER'].nunique().reset_index(name='failure_count')

    # Merge the two dataframes for df2
    merged_df2 = pd.merge(total_count_per_day_df2, failure_count_per_day_df2, on=date_column, how='left').fillna(0)

    # Convert the failure_count column to integer for df2
    merged_df2['failure_count'] = merged_df2['failure_count'].astype(int)

    # Calculate the Daily First Pass Yield (FPY) and convert it to percentage for df2
    merged_df2['FPY'] = ((merged_df2['total_count'] - merged_df2['failure_count']) / merged_df2['total_count']) * 100

    # Visualization for df2
    # Create a figure with a secondary y-axis for df2
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Add total_count as a bar chart for df2
    fig2.add_trace(
        go.Bar(x=merged_df2['Date'], y=merged_df2['total_count'], name='Total Count', marker_color='blue'),
        secondary_y=False,
    )

    # Add failure_count as a bar chart, overlaid on total_count for df2
    fig2.add_trace(
        go.Bar(x=merged_df2['Date'], y=merged_df2['failure_count'], name='Failure Count', marker_color='red'),
        secondary_y=False,
    )

    # Update the layout for df2
    fig2.update_layout(
        title_text="2K First Pass Yield (FPY) and Counts",
        xaxis_title="Date",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        barmode='overlay'
    )

    # Add FPY as a line chart on the secondary y-axis for df2
    fig2.add_trace(
        go.Scatter(
            x=merged_df2['Date'],
            y=merged_df2['FPY'],
            name='First Pass Yield (FPY)',
            mode='lines+markers',
            line=dict(color='green')
        ),
        secondary_y=True,
    )

    # Set y-axes titles for df2
    fig2.update_yaxes(title_text="Count", secondary_y=False)
    fig2.update_yaxes(title_text="First Pass Yield (FPY) (%)", secondary_y=True)

    # Streamlit app layout for df2
    st.title("2K")
    st.plotly_chart(fig2, use_container_width=True)

    ############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(80)

    ########################2K heatmap

    # Grouping by 'Zone' and listing all 'SERIAL_NUMBER' in that group, along with the count of occurrences
    grouped_df_zone = filtered_df_10.groupby('Zone').agg({'SERIAL_NUMBER': list, 'Height': 'count'}).reset_index()
    grouped_df_zone = grouped_df_zone.rename(columns={'Height': 'Count'})

    # Sorting the dataframe by 'Count' in descending order
    sorted_df2 = grouped_df_zone .sort_values(by='Count', ascending=False)

    # Use the rename() method to change the column name
    sorted_df2.rename(columns={'Count': 'Failing_times'}, inplace=True)

    # Convert the "Zone" column to an integer type
    sorted_df2['Zone'] = sorted_df2['Zone'].astype(int)

    # Coordinates data
    coordinates_df2 = pd.DataFrame({
        'shapeId': [1, 2, 3, 4, 5],
        'pointX': [460, 462, 464, 457,460],
        'pointY': [36, 171, 281, 396, 529]
    })


    ## Convert 'Failing_times' to integer if it's not already
    sorted_df2['Failing_times'] = sorted_df2['Failing_times'].astype(int)

    # Merge the dataframes
    merged_df2 = pd.merge(sorted_df2, coordinates_df2, left_on='Zone', right_on='shapeId')

    # Path to the uploaded image file
    # NOTE: You'll need to change the path to where your image is located on the server or load it differently
    image_path = r"C:\Users\hollee\Desktop\Denver\32K\raw data dashboard\32K_2K.jpg"

    # Read the image file and encode it in base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Create a bubble plot with red bubbles and display 'Failing_times' on hover
    fig = px.scatter(
        merged_df2,
        x='pointX',
        y='pointY',
        size='Failing_times',
        color_discrete_sequence=['red'],
        hover_name='Failing_times',
        hover_data={'pointX': False, 'pointY': False}
    )

    # Adjust the text position
    fig.update_traces(textposition='top center')

    # Get the dimensions of the image
    img_width, img_height = 1044, 694

    # Use the encoded image string as source
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{encoded_string}',
            xref="x",
            yref="y",
            x=0,
            y=img_height,
            sizex=img_width,
            sizey=img_height,
            sizing="contain",
            opacity=1.0,
            layer="below"
        )
    )

    # Update axes ranges to fit image
    fig.update_xaxes(range=[0, img_width])
    fig.update_yaxes(range=[0, img_height])

    # Update the layout to remove axes and gridlines
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        width=img_width,
        height=img_height,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # In Streamlit, display the figure with its original dimensions
    st.plotly_chart(fig, use_container_width=False)

############################################################################## Update progress to indicate pre-query operations
    progress_bar.progress(85)
    ###################### 2K Zone failure

    # Grouping by 'Zone' and listing all 'SERIAL_NUMBER' in that group, along with the count of occurrences
    grouped_df_zone = filtered_df_10.groupby('Zone').agg({'SERIAL_NUMBER': list, 'Height': 'count'}).reset_index()
    grouped_df_zone = grouped_df_zone.rename(columns={'Height': 'Count'})

    # Sorting the dataframe by 'Count' in descending order
    sorted_df2 = grouped_df_zone .sort_values(by='Count', ascending=False)

    # Use the rename() method to change the column name
    sorted_df2.rename(columns={'Count': 'Failing_times'}, inplace=True)

    # Convert the "Zone" column to an integer type
    sorted_df2['Zone'] = sorted_df2['Zone'].astype(int)

    # Convert the integer "Zone" values to string without the decimal
    sorted_df2['Zone'] = sorted_df2['Zone'].astype(str)
    # Assuming 'sorted_df2' is your second DataFrame

    # Convert 'Zone' to string to maintain order in sorted_df2
    sorted_df2['Zone'] = sorted_df2['Zone'].astype(str)

    # Get unique SERIAL_NUMBERS for sorted_df2
    serial_numbers_df2 = sorted_df2['SERIAL_NUMBER'].explode().unique()

    # Create a color scale for SERIAL_NUMBERS in sorted_df2
    colors_df2 = px.colors.qualitative.Set1
    color_dict_df2 = {serial: colors_df2[i % len(colors_df2)] for i, serial in enumerate(serial_numbers_df2)}

    # Initialize a Plotly figure for sorted_df2
    fig_df2 = go.Figure()

    # Add bars for each SERIAL_NUMBER in sorted_df2
    for serial in serial_numbers_df2:
        serial_counts_df2 = sorted_df2.apply(lambda x: x['SERIAL_NUMBER'].count(serial), axis=1)
        fig_df2.add_trace(go.Bar(
            x=sorted_df2['Zone'],
            y=serial_counts_df2,
            name=str(serial),
            marker_color=color_dict_df2[serial]
        ))

    # Calculate the cumulative percentage for the Pareto line in sorted_df2
    sorted_df2['Cumulative_Percentage'] = sorted_df2['Failing_times'].cumsum() / sorted_df2['Failing_times'].sum() * 100

    # Add the Pareto line with white color for sorted_df2
    fig_df2.add_trace(go.Scatter(
        x=sorted_df2['Zone'],
        y=sorted_df2['Cumulative_Percentage'],
        name='Cumulative Percentage',
        mode='lines+markers',
        marker=dict(color='white'),
        line=dict(color='white'),
        yaxis='y2'
    ))

    # Update the layout for a stacked bar chart with a secondary y-axis for the Pareto line in sorted_df2
    fig_df2.update_layout(
        barmode='stack',
        title='2K Zone Failure Pareto Analysis)',
        xaxis_title='Zone',
        yaxis_title='Failing Times',
        yaxis2=dict(
            title='Cumulative Percentage',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 110]
        ),
        legend_title='SERIAL_NUMBER',
        xaxis={'type': 'category'},
        legend=dict(
            x=1.05,
            y=1,
            orientation="v"
        )
    )

    # Streamlit app layout for sorted_df2
    st.plotly_chart(fig_df2, use_container_width=True)

        # Update progress to indicate post-query operations
    time.sleep(0.5)  # Simulated delay
    progress_bar.progress(100)
