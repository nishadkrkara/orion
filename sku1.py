import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from hijri_converter import convert
import holidays

# Load data
# Replace 'final_file.csv' with the path to your dataset
df = pd.read_csv('final_file.csv', parse_dates=['INVOICE_DT'])

# Ensure the necessary columns are present
required_columns = ['INVOICE_DT', 'STORE_NAME', 'SKU_CODE', 'COUNTRY_CODE', 'BRAND_CODE', 'NET_SALES_AMOUNT']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in the dataset.")
        st.stop()

# Streamlit App
st.title('Sales Forecast Dashboard')

# 1. Select Country
selected_country = st.selectbox('Select Country', sorted(df['COUNTRY_CODE'].unique()))

# Filter dataframe based on selected country
df_country = df[df['COUNTRY_CODE'] == selected_country]

# 2. Select Brand based on Country
available_brands = sorted(df_country['BRAND_CODE'].unique())
selected_brands = st.multiselect('Select Brand(s)', available_brands)

if selected_brands:
    df_brand = df_country[df_country['BRAND_CODE'].isin(selected_brands)]
else:
    df_brand = df_country.copy()

# 3. Select SKU based on selected Brands
available_skus = sorted(df_brand['SKU_CODE'].unique())
selected_skus = st.multiselect('Select SKU(s)', available_skus)

if selected_skus:
    df_sku = df_brand[df_brand['SKU_CODE'].isin(selected_skus)]
else:
    df_sku = df_brand.copy()

# 4. Select Store based on selected SKUs
available_stores = sorted(df_sku['STORE_NAME'].unique())
selected_stores = st.multiselect('Select Store(s)', available_stores)

if selected_stores:
    df_filtered = df_sku[df_sku['STORE_NAME'].isin(selected_stores)]
else:
    df_filtered = df_sku.copy()

# Define functions to get holidays
def get_islamic_holidays(start_year, end_year, country_code):
    islamic_holidays = []
    for year in range(start_year, end_year + 1):
        # Adjust year for Hijri conversion
        hijri_year = year - 579  # Approximate conversion
        
        # Ramadan (start date)
        ramadan_start = convert.Hijri(hijri_year, 9, 1).to_gregorian()
        # Eid al-Fitr
        eid_al_fitr = convert.Hijri(hijri_year, 10, 1).to_gregorian()
        # Eid al-Adha
        eid_al_adha = convert.Hijri(hijri_year, 12, 10).to_gregorian()

        islamic_holidays.extend([
            (ramadan_start, 'Ramadan Start'),
            (eid_al_fitr, 'Eid al-Fitr'),
            (eid_al_adha, 'Eid al-Adha')
        ])
    return islamic_holidays

def get_black_friday(years):
    black_fridays = []
    for year in years:
        # Black Friday is the fourth Friday in November
        november = pd.date_range(start=f'{year}-11-01', end=f'{year}-11-30', freq='D')
        fridays = november[november.weekday == 4]
        if len(fridays) >= 4:
            black_friday = fridays[3]  # Fourth Friday
            black_fridays.append((black_friday, 'Black Friday'))
    return black_fridays

# Data preparation
def prepare_data(df_filtered):
    # Copy df to avoid modifying the original
    df_copy = df_filtered.copy()
    
    # Get the last date in the dataset
    last_date = df_copy['INVOICE_DT'].max()
    last_year = last_date.year

    # Calculate the date two years before the last date
    start_date = last_date - pd.DateOffset(years=2)
    start_year = start_date.year

    # Filter the dataframe for the last two years
    df_last_two_years = df_copy[df_copy['INVOICE_DT'] >= start_date]
    
    # Check if there is data after filtering
    if df_last_two_years.empty:
        return None

    # Add 'Holiday' column
    df_last_two_years['Holiday'] = 'None'

    # Get Islamic holidays
    islamic_holidays = get_islamic_holidays(start_year, last_year, selected_country)
    # Get Black Friday dates
    black_fridays = get_black_friday(range(start_year, last_year + 1))

    # Combine all holidays
    all_holidays = islamic_holidays + black_fridays

    # Create a DataFrame for holidays
    holidays_df = pd.DataFrame(all_holidays, columns=['Date', 'Holiday_Name'])
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])

    # Merge holidays into the sales data
    df_last_two_years = df_last_two_years.merge(holidays_df, left_on='INVOICE_DT', right_on='Date', how='left')
    df_last_two_years['Holiday'] = df_last_two_years['Holiday_Name'].fillna('None')
    df_last_two_years.drop(['Date', 'Holiday_Name'], axis=1, inplace=True)

    # Add 'Season' column based on month
    df_last_two_years['Season'] = df_last_two_years['INVOICE_DT'].dt.month % 12 // 3 + 1
    df_last_two_years['Season'] = df_last_two_years['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'})

    # Compute daily averages including holidays and seasons
    daily_averages = df_last_two_years.groupby([
        'STORE_NAME', 'SKU_CODE', 'COUNTRY_CODE', 'BRAND_CODE', 'Holiday', 'Season'
    ])['NET_SALES_AMOUNT'].mean().reset_index()
    daily_averages.rename(columns={'NET_SALES_AMOUNT': 'SALES'}, inplace=True)

    # Generate future dates for the next 6 months
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(days=1),
        periods=180 ,  # 6 months * ~30 days
        freq='D'  # Daily frequency
    )
    forecast_df = pd.DataFrame({'INVOICE_DT': future_dates})

    # Assign holidays to future dates
    forecast_df['Holiday'] = 'None'
    future_years = range(last_year, future_dates.max().year + 1)
    future_islamic_holidays = get_islamic_holidays(last_year, future_dates.max().year, selected_country)
    future_black_fridays = get_black_friday(future_years)
    future_all_holidays = future_islamic_holidays + future_black_fridays
    future_holidays_df = pd.DataFrame(future_all_holidays, columns=['Date', 'Holiday_Name'])
    future_holidays_df['Date'] = pd.to_datetime(future_holidays_df['Date'])
    forecast_df = forecast_df.merge(future_holidays_df, left_on='INVOICE_DT', right_on='Date', how='left')
    forecast_df['Holiday'] = forecast_df['Holiday_Name'].fillna('None')
    forecast_df.drop(['Date', 'Holiday_Name'], axis=1, inplace=True)

    # Add 'Season' to future dates
    forecast_df['Season'] = forecast_df['INVOICE_DT'].dt.month % 12 // 3 + 1
    forecast_df['Season'] = forecast_df['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'})

    # Get unique combinations of group keys
    unique_groups = df_last_two_years[['STORE_NAME', 'SKU_CODE', 'COUNTRY_CODE', 'BRAND_CODE']].drop_duplicates()

    if unique_groups.empty:
        return None

    # Create a cartesian product of unique groups and forecast dates
    forecast_groups = unique_groups.assign(key=1).merge(forecast_df.assign(key=1), on='key').drop('key', axis=1)

    # Merge with daily averages to get the forecasted sales
    forecast = forecast_groups.merge(
        daily_averages,
        on=['STORE_NAME', 'SKU_CODE', 'COUNTRY_CODE', 'BRAND_CODE', 'Holiday', 'Season'],
        how='left'
    )

    # Fill any missing SALES with overall average
    overall_average = df_last_two_years['NET_SALES_AMOUNT'].mean()
    forecast['SALES'].fillna(overall_average, inplace=True)

    return forecast

# Prepare the data
forecast = prepare_data(df_filtered)

if forecast is None or forecast.empty:
    st.write("No data available for the selected options.")
else:
    # Sum SALES over the selected groups for each date
    forecast_summary = forecast.groupby('INVOICE_DT')['SALES'].sum().reset_index()

    # Plot the forecast
    st.subheader('Forecast for the Next 6 Months')
    st.line_chart(forecast_summary.set_index('INVOICE_DT'))

    # Display the forecast data
    st.write('Forecast Data:')
    st.dataframe(forecast_summary)