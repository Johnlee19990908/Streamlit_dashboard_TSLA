import os
import clickhouse_connect
import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objs as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")
st.title("32K Coherix Raw Data Dashboard")


client = clickhouse_connect.get_client(host='xxxxxxxxxx', 

                        port=8123, username='xxxxxxxxxxxx', 

                        password='xxxxxxxxxxxxx',

                        query_limit=0)
query = '''
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
		    and KAFKA_TIMESTAMP> now() - interval 30 day
    )
    AND TRACE_TAG IN ('Height', 'Width', 'Zone')
'''
df = client.query_df(query)


# Filter the DataFrame for TRACE_TAG = 'Height'
height_df = df[df['TRACE_TAG'] == 'Height']

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
width_df = df[df['TRACE_TAG'] == 'Width']

# Initialize an empty DataFrame to hold the expanded data for Width
expanded_width_data = pd.DataFrame()

# Iterate over each row in the width_df
for index, row in width_df.iterrows():
    # Convert the TRACE_VALUES list into a DataFrame with a 'Width' column
    temp_df = pd.DataFrame(row['TRACE_VALUES'], columns=["Width"])

    # Append this temp_df to the expanded_width_data DataFrame
    expanded_width_data = pd.concat([expanded_width_data, temp_df], ignore_index=True)


# Filter the DataFrame for TRACE_TAG = 'Zone'
zone_df = df[df['TRACE_TAG'] == 'Zone']

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

################################################renew code
# Calculate the maximum index for each SERIAL_NUMBER in merged_df2
max_indices = merged_df.groupby('SERIAL_NUMBER')['SERIAL_NUMBER_index'].max()

# Find SERIAL_NUMBERs with max index > 13000
serial_numbers_to_remove2 = max_indices[max_indices > 13000].index

# Filter out rows in merged_df2 where SERIAL_NUMBER_index > 11505 and SERIAL_NUMBER is in the list of serial_numbers_to_remove2
filtered_merged_df = merged_df[~((merged_df['SERIAL_NUMBER'].isin(serial_numbers_to_remove2)) & (merged_df['SERIAL_NUMBER_index'] > 11505))]

# Convert KAFKA_TIMESTAMP to datetime if it's not already
filtered_merged_df['KAFKA_TIMESTAMP'] = pd.to_datetime(filtered_merged_df['KAFKA_TIMESTAMP'])

# Extract the date part and create a new column 'Date'
filtered_merged_df['Date'] = filtered_merged_df['KAFKA_TIMESTAMP'].dt.date
filtered_merged_df['Date'] = pd.to_datetime(filtered_merged_df['Date'])

#########################################################renew code end here



#########################visualize

# Create a text input widget for serial number
serial_number_input = st.text_input("Enter SERIAL_NUMBER to filter")

# Function to filter two DataFrames based on serial number
def filter_by_serial_number(df1, df2, serial_number):
    if serial_number:
        return df1[df1['SERIAL_NUMBER'] == serial_number], df2[df2['SERIAL_NUMBER'] == serial_number]
    return df1, df2
# Apply the filter to both dataframes
filtered_merged_df, filtered_data_2k = filter_by_serial_number(filtered_merged_df, filtered_merged_df, serial_number_input)


min_date = filtered_merged_df['Date'].min().date()
max_date = filtered_merged_df['Date'].max().date()

# Assuming max_date is defined as the latest date in your dataset
max_date = filtered_merged_df['Date'].max()
min_date = max_date - pd.Timedelta(days=1)  # Set min_date to one day before max_date

# Shared date input for both dashboards
start_date, end_date = st.date_input("Select Date Range", [min_date, max_date], key='shared_date_range')

# Convert start_date and end_date to Pandas Timestamps
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

filtered_data_rtv = filtered_merged_df[
    (filtered_merged_df['Date'] >= start_date) & 
    (filtered_merged_df['Date'] <= end_date)
]


# Define color palette
colors = px.colors.qualitative.Plotly

# Rest of your Streamlit code
st.title("RTV Dispense")

# Plotting the data
# Create the scatter plot for Height
fig_height = px.scatter(filtered_data_rtv, x='SERIAL_NUMBER_index', y='Height', 
                        color='SERIAL_NUMBER', color_discrete_sequence=colors)

# Add min and max limit lines for Height
fig_height.add_traces([
    go.Scatter(x=filtered_data_rtv ['SERIAL_NUMBER_index'], y=filtered_data_rtv ['Height min limit'], 
               mode='lines', name='Height Min Limit', line=dict(color='red')),
    go.Scatter(x=filtered_data_rtv ['SERIAL_NUMBER_index'], y=filtered_data_rtv ['Height max limit'], 
               mode='lines', name='Height Max Limit', line=dict(color='red'))
])


# Update the layout for a dark background
fig_height.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')

# Display the graph in Streamlit
st.plotly_chart(fig_height, use_container_width=True)

# Create the scatter plot for Width using the same filtered_data
fig_width = px.scatter(filtered_data_rtv , x='SERIAL_NUMBER_index', y='Width', 
                       color='SERIAL_NUMBER', color_discrete_sequence=colors)

# Add min and max limit lines for Width
fig_width.add_traces([
    go.Scatter(x=filtered_data_rtv ['SERIAL_NUMBER_index'], y=filtered_data_rtv ['Width min limit'], 
               mode='lines', name='Width Min Limit', line=dict(color='red')),
    go.Scatter(x=filtered_data_rtv ['SERIAL_NUMBER_index'], y=filtered_data_rtv ['Width max limit'], 
               mode='lines', name='Width Max Limit', line=dict(color='red'))
])


# Update the layout for a dark background
fig_width.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')


# Display the graph in Streamlit
st.plotly_chart(fig_width, use_container_width=True)



###################################2K
query = '''
SELECT
    TRACE_TAG,
    SERIAL_NUMBER,
    PROCESS_RESULT,
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
		    and KAFKA_TIMESTAMP> now() - interval 30 day
    )
    AND TRACE_TAG IN ('Height', 'Width', 'Zone')
'''
df2 = client.query_df(query)

# Assuming df2 is your DataFrame, filter it for TRACE_TAG = 'Height'
height_df2 = df2[df2['TRACE_TAG'] == 'Height']

# Initialize an empty DataFrame to hold the expanded data
expanded_data2 = pd.DataFrame()

# Iterate over each row in the height_df2
for index, row in height_df2.iterrows():
    # Convert the TRACE_VALUES list into a DataFrame
    temp_df2 = pd.DataFrame(row['TRACE_VALUES'], columns=["Height"])

    # Add SERIAL_NUMBER and KAFKA_TIMESTAMP columns to this DataFrame
    temp_df2['SERIAL_NUMBER'] = row['SERIAL_NUMBER']
    temp_df2['KAFKA_TIMESTAMP'] = row['KAFKA_TIMESTAMP']

    # Append this temp_df2 to the expanded_data2 DataFrame
    expanded_data2 = pd.concat([expanded_data2, temp_df2], ignore_index=True)
    
# Adding a 'SERIAL_NUMBER index' column that counts the index within each SERIAL_NUMBER group
expanded_data2['SERIAL_NUMBER_index'] = expanded_data2.groupby('SERIAL_NUMBER').cumcount()

# Filter the DataFrame for TRACE_TAG = 'Width'
width_df2 = df2[df2['TRACE_TAG'] == 'Width']

# Initialize an empty DataFrame to hold the expanded data for Width
expanded_width_data2 = pd.DataFrame()

# Iterate over each row in the width_df2
for index, row in width_df2.iterrows():
    # Convert the TRACE_VALUES list into a DataFrame with a 'Width' column
    temp_df2 = pd.DataFrame(row['TRACE_VALUES'], columns=["Width"])

    # Append this temp_df2 to the expanded_width_data2 DataFrame
    expanded_width_data2 = pd.concat([expanded_width_data2, temp_df2], ignore_index=True)


# Filter the DataFrame for TRACE_TAG = 'Zone'
zone_df2 = df2[df2['TRACE_TAG'] == 'Zone']

# Initialize an empty DataFrame to hold the expanded data for Zone
expanded_zone_data2 = pd.DataFrame()

# Iterate over each row in the zone_df2
for index, row in zone_df2.iterrows():
    # Convert the TRACE_VALUES list into a DataFrame with a 'Zone' column
    temp_df2 = pd.DataFrame(row['TRACE_VALUES'], columns=["Zone"])

    # Append this temp_df2 to the expanded_zone_data2 DataFrame
    expanded_zone_data2 = pd.concat([expanded_zone_data2, temp_df2], ignore_index=True)

# Define the height limits for each zone
height_limits_2K = {
    1: {'max': 10.5, 'min': 3.5},
    2: {'max': 10.5, 'min': 3},
    3: {'max': 10.5, 'min': 3},
    4: {'max': 10, 'min': 2.75},
    5: {'max': 10, 'min': 2.75},
}

# Define the width limits for each zone
width_limits_2K = {
    1: {'max': 9, 'min': 5},
    2: {'max': 10, 'min': 4},
    3: {'max': 10, 'min': 4},
    4: {'max': 10, 'min': 4},
    5: {'max': 10, 'min': 4},
}

# Add the min and max limit columns for Height
expanded_zone_data2['Height min limit'] = expanded_zone_data2['Zone'].map(lambda z: height_limits_2K.get(z, {}).get('min', None))
expanded_zone_data2['Height max limit'] = expanded_zone_data2['Zone'].map(lambda z: height_limits_2K.get(z, {}).get('max', None))

# Add the min and max limit columns for Width
expanded_zone_data2['Width min limit'] = expanded_zone_data2['Zone'].map(lambda z: width_limits_2K.get(z, {}).get('min', None))
expanded_zone_data2['Width max limit'] = expanded_zone_data2['Zone'].map(lambda z: width_limits_2K.get(z, {}).get('max', None))

# Concatenate the three DataFrames along the columns
merged_df2 = pd.concat([expanded_data2, expanded_width_data2, expanded_zone_data2], axis=1)

################################################renew code
# Calculate the maximum index for each SERIAL_NUMBER in merged_df2
max_indices2 = merged_df2.groupby('SERIAL_NUMBER')['SERIAL_NUMBER_index'].max()

# Find SERIAL_NUMBERs with max index > 13000
serial_numbers_to_remove2 = max_indices2[max_indices2 > 13000].index

# Filter out rows in merged_df2 where SERIAL_NUMBER_index > 11505 and SERIAL_NUMBER is in the list of serial_numbers_to_remove2
filtered_merged_df2 = merged_df2[~((merged_df2['SERIAL_NUMBER'].isin(serial_numbers_to_remove2)) & (merged_df2['SERIAL_NUMBER_index'] > 11505))]

# Convert KAFKA_TIMESTAMP to datetime if it's not already
filtered_merged_df2['KAFKA_TIMESTAMP'] = pd.to_datetime(filtered_merged_df2['KAFKA_TIMESTAMP'])

# Extract the date part and create a new column 'Date'
filtered_merged_df2['Date'] = filtered_merged_df2['KAFKA_TIMESTAMP'].dt.date
#########################################################renew code end here

# Rest of your Streamlit code
st.title("2K Dispense")
# Apply the filter to both dataframes
filtered_data1, filtered_merged_df2 = filter_by_serial_number(filtered_merged_df, filtered_merged_df2, serial_number_input)


filtered_merged_df2['Date'] = pd.to_datetime(filtered_merged_df2['Date'])

filtered_data_2k = filtered_merged_df2[
    (filtered_merged_df2['Date'] >= start_date) & 
    (filtered_merged_df2['Date'] <= end_date)
]

# Define color palette
colors = px.colors.qualitative.Plotly

filtered_data_2k.sort_values("SERIAL_NUMBER_index", inplace=True)

# Create the scatter plot for Height
fig_height = px.scatter(filtered_data_2k, x='SERIAL_NUMBER_index', y='Height', 
                        color='SERIAL_NUMBER', color_discrete_sequence=colors)

# Add min and max limit lines for Height
fig_height.add_traces([
    go.Scatter(x=filtered_data_2k['SERIAL_NUMBER_index'], y=filtered_data_2k['Height min limit'], 
               mode='lines', name='Height Min Limit', line=dict(color='red')),
    go.Scatter(x=filtered_data_2k['SERIAL_NUMBER_index'], y=filtered_data_2k['Height max limit'], 
               mode='lines', name='Height Max Limit', line=dict(color='red'))
])

# Create the scatter plot for Width using the same filtered_data
fig_width = px.scatter(filtered_data_2k, x='SERIAL_NUMBER_index', y='Width', 
                       color='SERIAL_NUMBER', color_discrete_sequence=colors)

# Add min and max limit lines for Width
fig_width.add_traces([
    go.Scatter(x=filtered_data_2k['SERIAL_NUMBER_index'], y=filtered_data_2k['Width min limit'], 
               mode='lines', name='Width Min Limit', line=dict(color='red')),
    go.Scatter(x=filtered_data_2k['SERIAL_NUMBER_index'], y=filtered_data_2k['Width max limit'], 
               mode='lines', name='Width Max Limit', line=dict(color='red'))
])

# Update the layout for both figures for a dark background
for fig in [fig_height, fig_width]:
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')

# Display the graphs
st.plotly_chart(fig_height, use_container_width=True)
st.plotly_chart(fig_width, use_container_width=True)



