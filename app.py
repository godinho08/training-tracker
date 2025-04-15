import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import io
from urllib.request import urlopen

import calplot




# Set page configuration
st.set_page_config(
    page_title="Training Tracker Dashboard",
    page_icon="🏋️",
    layout="wide"
)

# Define the URL of the data
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTyrIt51KdAy6obZunt9_epHauWZLpLcITtL8ZQY41ZICQ2iKOYxNI-y5v7axoAM-rCd8n7YBfiRE6W/pub?gid=1788754157&single=true&output=csv"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 1rem;
        color: #777;
    }
    .date-filters-container {
        display: flex;
        gap: 20px;
        align-items: flex-start;
    }
    .date-filter-column {
        flex: 0 0 auto;
        padding: 15px;
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

#Hide the modebar from plotly charts
def hide_modebar(fig):
    return st.plotly_chart(fig, config={'displayModeBar': False})

# Load the data
@st.cache_data
def load_data():
    """Load and prepare the training data."""
    # Read the data
    df = pd.read_csv(DATA_URL)
    
    # Convert dates and filter future dates immediately
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] <= pd.Timestamp.today()]  # Filter out future dates first
    
    # Fill NaN values in training_type with 'Rest'
    df['training_type'] = df['training_type'].fillna('Rest')
    
    # Create derived features
    df['weekday'] = df['Date'].dt.day_name()
    df['week_number'] = df['Date'].dt.isocalendar().week
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['month_year'] = df['Date'].dt.strftime('%b %Y')
    df['training_duration'] = 1  # Assuming each training session is 1 unit
    
    # Assign consistent colors to training types
    unique_types = df['training_type'].unique()
    colors = px.colors.qualitative.Bold[:len(unique_types)]  # Get n distinct colors
    color_map = dict(zip(unique_types, colors))
    df['type_color'] = df['training_type'].map(color_map)
    
    return df

# Define a consistent color mapping for training types
TRAINING_COLORS = {
    'Weights': '#1E88E5',  # Blue
    'Run': '#FFC107',      # Amber
    'Bike': '#4CAF50',     # Green
    'Swim': '#9C27B0',     # Purple
    'Yoga': '#FF9800',     # Orange
    'Rest': '#757575',     # Gray
    'Other': '#607D8B'     # Blue Gray
}

def get_color_sequence(df):
    """Get color sequence for the unique training types in the data."""
    unique_types = df['training_type'].unique()
    return [TRAINING_COLORS.get(t, TRAINING_COLORS['Other']) for t in unique_types]

#--- Get Modified Dataframes ---
def tooltip_manager(df, view):
    """Create tooltips for the training data."""
    if view == 'monthly view':
        tooltip = (
            '<span style="color: grey; font-size: 12px;">' + df['start_date'] + ' - ' + df['end_date'] + '</span>'
            + '<br><span style="color: black; font-size: 15px;">' + df['sum'].astype(str) + ' sessions</span>'
        )
    elif view == 'weekly view':
        tooltip = (
            '<span style="color: grey; font-size: 12px;">' + df['start_date'] + ' - ' + df['end_date'] + '</span>'
            + '<br><span style="color: black; font-size: 15px;">' + df['sum'].astype(str) + ' sessions</span>'
        )
    else:
        # error handling
        st.error("Invalid view type for tooltip manager.")
        return None
        
    return tooltip

def get_monthly_counts_df(df):
    """Get monthly counts of training sessions."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    monthly_counts = df.groupby(df['Date'].dt.to_period('M'))['Trained (1/0)'].agg(
        sum='sum', count='count'
    )
    monthly_counts['start_date'] = df.groupby(df['Date'].dt.to_period('M'))['Date'].min().dt.strftime('%b %d')
    monthly_counts['end_date'] = df.groupby(df['Date'].dt.to_period('M'))['Date'].max().dt.strftime('%b %d')
    
    # Convert periods to strings before resetting the index
    monthly_counts = monthly_counts.reset_index()
    monthly_counts['month'] = monthly_counts['Date'].astype(str)
    monthly_counts = monthly_counts.drop('Date', axis=1)

    return monthly_counts

def get_weekly_counts_df(df):
    """Get weekly counts of training sessions."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group by week
    weekly_counts = df.groupby(df['Date'].dt.to_period('W'))['Trained (1/0)'].agg(
        sum='sum', count='count'
    )
    weekly_counts['start_date'] = df.groupby(df['Date'].dt.to_period('W'))['Date'].min().dt.strftime('%b %d')
    weekly_counts['end_date'] = df.groupby(df['Date'].dt.to_period('W'))['Date'].max().dt.strftime('%b %d')
    
    # Convert periods to strings before resetting the index
    weekly_counts = weekly_counts.reset_index()
    weekly_counts['week'] = weekly_counts['Date'].astype(str)
    weekly_counts = weekly_counts.drop('Date', axis=1)
    
    return weekly_counts
    
# Function to create the dashboard
def create_dashboard(df, target_ratio):
    """Create the main dashboard."""
    # Title and description
    st.markdown('<div class="main-header">🏋️ Training Tracker</div>', unsafe_allow_html=True)
    #st.write("📈 Analyzing my training patterns and progress over time.")

    # Create a container for date selection with custom CSS for better fit
    st.markdown("""
    <style>
    .date-filters-container {
        display: flex;
        gap: 20px;
        align-items: flex-start;
    }
    .date-filter-column {
        flex: 0 0 auto;
        padding: 15px;
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)
    

    # Get data for date filters
    current_year = pd.Timestamp.today().year
    month_names = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
    current_month = pd.Timestamp.today().month
    
    # Get months that have already passed this year
    passed_months = month_names[:current_month]
    
    # Create options list with current year and passed months
    all_options = [str(current_year)] + passed_months
    
    # Create a single pills component for all options
    selected_option = st.pills("Time Period", options=all_options, default=str(current_year))
    
    # Handle the selection based on the option type
    if selected_option in month_names:
        # Month view
        selected_month_num = month_names.index(selected_option) + 1
        selected_year = current_year
        
        # Filter for specific month and year
        start_date = pd.Timestamp(year=selected_year, month=selected_month_num, day=1)
        # Calculate last day of month
        if selected_month_num == 12:
            end_date = pd.Timestamp(year=selected_year, month=12, day=31)
        else:
            end_date = pd.Timestamp(year=selected_year, month=selected_month_num+1, day=1) - timedelta(days=1)
        
        # Filter data for selected month
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
    else:  # Year view
        # Filter data for selected year
        selected_year = int(selected_option)
        start_date = pd.Timestamp(year=selected_year, month=1, day=1)
        end_date = pd.Timestamp(year=selected_year, month=12, day=31)
        
        # Filter data for selected year
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    
    # Display key metrics
    display_key_metrics(filtered_df)

    # Main visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Training Types", "Weekly Patterns", "Raw Data"])
    
    with tab1:
        display_training_overview(filtered_df, target_ratio)
    
    with tab2:
        display_training_types(filtered_df)
    
    with tab3:
        display_weekly_patterns(filtered_df)
    with tab4:
        display_raw_data(filtered_df)

# Functions for different dashboard sections
def display_key_metrics(df):
    """Display key performance metrics."""
    total_days = len(df)
    training_days = df['Trained (1/0)'].sum()
    training_percentage = (training_days / total_days) * 100 if total_days > 0 else 0
    
    # Calculate streak
    current_streak, longest_streak = calculate_streaks(df)
    
    # Get most common training type
    training_counts = df[df['training_type'] != 'Rest']['training_type'].value_counts()
    most_common_type = training_counts.idxmax() if not training_counts.empty else "N/A"
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="card">
                <div class="metric-value">{} <span style="color: #aaa; font-size: 1.5rem">/ {}</span></div>
                <div class="metric-label">Training Sessions</div>
            </div>
        """.format(training_days, total_days), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Training Consistency</div>
            </div>
        """.format(training_percentage), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Longest Streak (days)</div>
            </div>
        """.format(longest_streak), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Favorite Training</div>
            </div>
        """.format(most_common_type), unsafe_allow_html=True)

def calculate_streaks(df):
    """Calculate current and longest training streaks."""
    # Sort dataframe by date
    df_sorted = df.sort_values('Date')
    
    # Initialize streak counters
    current_streak = 0
    longest_streak = 0
    temp_streak = 0
    
    # Calculate streaks
    for trained in df_sorted['Trained (1/0)']:
        if trained == 1:
            temp_streak += 1
            longest_streak = max(longest_streak, temp_streak)
        else:
            temp_streak = 0
    
    # Calculate current streak (from the most recent days)
    for trained in df_sorted['Trained (1/0)'].iloc[::-1]:
        if trained == 1:
            current_streak += 1
        else:
            break
    
    return current_streak, longest_streak

## Make this different in the future



def display_training_overview(df, target_ratio):
    
    ### --- Compute Metrics
    df_sorted = df.copy()
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    df_sorted['Day_Number'] = range(1, len(df_sorted) + 1)
    df_sorted['Cum_Training_Days'] = df_sorted['Trained (1/0)'].cumsum()
    df_sorted['Cum_Training_Days_Target'] = df_sorted['Day_Number'] * target_ratio
    df_sorted['Target_Line'] = df_sorted['Day_Number'] * target_ratio
    
    # Calculate ratio (higher is better)
    df_sorted['Ratio'] = df_sorted['Cum_Training_Days'] / df_sorted['Day_Number']
    df_sorted['Target_Ratio'] = target_ratio  # Constant target ratio
    
    #Montly ratio
    df_sorted['Year-Month'] = df_sorted['Date'].dt.to_period('M')
    df_sorted['day_of_month'] = df_sorted['Date'].dt.day
    df_sorted['monthly_training_cum'] = df_sorted.groupby('Year-Month')['Trained (1/0)'].cumsum()
    df_sorted['monthly_ratio'] = df_sorted['monthly_training_cum'] / df_sorted['day_of_month']
        
    # Calculate inversed ratio (lower is better)
    df_sorted['inv_Ratio'] =  df_sorted['Day_Number'] / df_sorted['Cum_Training_Days']
    df_sorted['inv_Target_Ratio'] = 1/target_ratio  # Constant target ratio
    
    # Calculate rolling 30-day metrics
    df_sorted['Rolling_30d_Training'] = df_sorted['Trained (1/0)'].rolling(window=30, min_periods=1).sum()
    df_sorted['Rolling_30d_Count'] = df_sorted['Trained (1/0)'].rolling(window=30, min_periods=1).count()
    df_sorted['Rolling_30d_Ratio'] = df_sorted['Rolling_30d_Training'] / df_sorted['Rolling_30d_Count']
    df_sorted['Rolling_30d_inv_Ratio'] = df_sorted['Rolling_30d_Count'] / df_sorted['Rolling_30d_Training'].replace(0, np.nan)
    
    # Calculate Montly cumulative sum
    
    # Add formatted date column
    df_sorted['formatted_date'] = df_sorted['Date'].dt.strftime('%b %d')
    
    #Tolltips 
    # Create tooltips for each line
    cumsum_tooltip = [
        f'<span style="color: grey; font-size: 12px">{date}</span><br><span style="color: black; font-size: 15px;">{cumsum:.0f} sessions</span>'
        for date, cumsum in zip(df_sorted['formatted_date'], df_sorted['Cum_Training_Days'].round(0))
    ]
    
    cumsum_target_tooltip = [
        f'<span style="color: grey; font-size: 12px">{date}</span><br><span style="color: black; font-size: 15px;">Target: {target:.0f}</span>'
        for date, target in zip(df_sorted['formatted_date'], df_sorted['Cum_Training_Days_Target'].round(0))
    ]
    
    ratio_tooltip = [
        f'<span style="color: grey; font-size: 12px">{date}</span><br><span style="color: black; font-size: 15px;">Ratio: {ratio:.2f}</span>'
        for date, ratio in zip(df_sorted['formatted_date'], df_sorted['Ratio'].round(2))
    ]
    
    monthly_ratio_tooltip = [
        f'<span style="color: grey; font-size: 12px">{date}</span><br><span style="color: black; font-size: 15px;">Monthly: {ratio:.2f}</span>'
        for date, ratio in zip(df_sorted['formatted_date'], df_sorted['monthly_ratio'].round(2))
    ]
    
    ratio_target_tooltip = [
        f'<span style="color: grey; font-size: 12px">{date}</span><br><span style="color: black; font-size: 15px;">Target: {target:.2f}</span>'
        for date, target in zip(df_sorted['formatted_date'], df_sorted['Target_Ratio'].round(2))
    ]
        
    ### ---
    
    #"""Display training overview charts."""
    st.markdown('<div class="sub-header">Training Frequency Over Time</div>', unsafe_allow_html=True)
    
    # Training frequency over time
    col1, col2 = st.columns([2, 2])
    
    
    with col1:
        tab1, tab2 = st.tabs(["Cumulative View", "Ratio View"])

        with tab1:
            # Create figure
            fig = go.Figure()
            
            # Add target line
            fig.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=df_sorted['Cum_Training_Days_Target'],
                name='Target',
                line=dict(color='rgba(255, 165, 0, 0.8)', dash='dot'),
                customdata=cumsum_target_tooltip,
                hovertemplate='%{customdata}<extra></extra>',
                mode='lines+markers',
                marker=dict(
                    size=8,
                    opacity=0,
                    showscale=False,
                    color='rgba(255, 165, 0, 0.8)'
                )
            ))
            
            # Add cumulative training line
            fig.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=df_sorted['Cum_Training_Days'],
                name='Cumulative',
                line=dict(color='#1E88E5', shape='spline'),
                customdata=cumsum_tooltip,
                hovertemplate='%{customdata}<extra></extra>',
                mode='lines+markers',
                marker=dict(
                    size=8,
                    opacity=0.0,  # Make markers slightly visible by default
                    showscale=False,
                    color='#1E88E5'
                ),
                hoveron='points+fills',  # Enable hover on points
                hoverinfo='all'
            ))
            
            fig.update_layout(
                title='Number of Sessions',
                height=400,
                yaxis=dict(
                    showspikes = True, 
                    spikethickness=1, 
                    range=[0, max(df_sorted['Cum_Training_Days_Target'].max(), df_sorted['Cum_Training_Days'].max() ) * 1.1],
                    title='Count'
                ),
                xaxis=dict(
                    showspikes=True,
                    spikethickness=1,
                    title='Date'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='closest',
                hoverdistance=100
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            # Create figure
            fig = go.Figure()
            # Add target line as a scatter trace instead of hline
            fig.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=[target_ratio] * len(df_sorted),
                name='Target',
                line=dict(color='rgba(255, 165, 0, 0.8)', dash='dot'),
                customdata=ratio_target_tooltip,
                hovertemplate='%{customdata}<extra></extra>',
                mode='lines+markers',
                marker=dict(
                    size=8,
                    opacity=0,
                    showscale=False,
                    color='rgba(255, 165, 0, 0.8)'
                )
            ))
            
            # Add monthly ratio line
            fig.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=df_sorted['monthly_ratio'],
                name='Monthly Ratio',
                line=dict(dash='dot'),
                customdata=monthly_ratio_tooltip,
                hovertemplate='%{customdata}<extra></extra>',
                mode='lines+markers',
                marker=dict(
                    size=8,
                    opacity=0,
                    showscale=False,
                    color='#666666'
                )
            ))
            
            # Add ratio line
            fig.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=df_sorted['Ratio'],
                name='Overall Ratio',
                line=dict(color='#1E88E5', shape='spline'),
                customdata=ratio_tooltip,
                hovertemplate='%{customdata}<extra></extra>',
                mode='lines+markers',
                marker=dict(
                    size=8,
                    opacity=0,
                    showscale=False,
                    color='#1E88E5'
                )
            ))
            
            fig.update_layout(
                title='Training Ratio <span style="color: grey; font-weight: normal;">(Training Days / Total Days)</span>',
                height=400,
                yaxis=dict(range=[0, 1.1], showspikes = True, spikethickness=1),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                hovermode='closest',
                hoverdistance=100,
                xaxis=dict(
                    showspikes=True,
                    spikethickness=1
                )
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Monthly and weekly training counts
    with col2:
        monthly_counts = get_monthly_counts_df(df)
        weekly_counts = get_weekly_counts_df(df)
        
        monthly_counts['tooltip'] = tooltip_manager(monthly_counts, 'monthly view')
        weekly_counts['tooltip'] = tooltip_manager(weekly_counts, 'weekly view')
        
        tab1, tab2 = st.tabs(["Weekly View", "Monthly View"])
        
        # Weekly training counts
        with tab1:
            fig = px.bar(weekly_counts, x='start_date', y='sum', 
                        title='Weekly Training Count',
                        labels={'sum': 'Sessions', 'start_date': 'Week'})
            fig.update_traces(marker_color='#1E88E5', hovertemplate='%{customdata}', customdata=weekly_counts['tooltip'])
            fig.update_layout(
                height=400,
                xaxis=dict(
                    tickmode='array',
                    tickvals=weekly_counts['start_date'][::2],
                    ticktext=weekly_counts['start_date'][::2],
                    #showspikes=True,
                    #spikethickness=1,
                    title='Date'
                ),
                yaxis=dict(
                    #showspikes=True,
                    #spikethickness=1,
                    title='Count'
                ),
                hovermode='closest',
                hoverdistance=100
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        # Monthly training counts
        with tab2:
            fig = px.bar(monthly_counts, x='month', y='sum', 
                        title='Monthly Training Count',
                        labels={'sum': 'Sessions', 'month': 'Month'})
            fig.update_traces(marker_color='#1E88E5', hovertemplate='%{customdata}', customdata=monthly_counts['tooltip'])
            fig.update_layout(
                height=400,
                xaxis=dict(
                    #showspikes=True,
                    #spikethickness=1,
                    title='Date'
                ),
                yaxis=dict(
                    #showspikes=True,
                    #spikethickness=1,
                    title='Count'
                ),
                hovermode='closest',
                hoverdistance=100
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        
    
    
def display_training_types(df):
    """Display training type analysis."""
    st.markdown('<div class="sub-header">Training Type Analysis</div>', unsafe_allow_html=True)
    
    # Filter to only training days
    training_df = df[df['Trained (1/0)'] == 1]
    
    # Create color map from the dataframe
    color_map = dict(zip(training_df['training_type'], training_df['type_color']))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training type distribution
        training_counts = training_df['training_type'].value_counts().reset_index()
        training_counts.columns = ['Training Type', 'Count']
        
        # Create tooltip for pie chart
        pie_tooltip = [
            f'<span style="color: grey; font-size: 12px">{type_}</span><br><span style="color: black; font-size: 15px;">{count} sessions</span>'
            for type_, count in zip(training_counts['Training Type'], training_counts['Count'])
        ]
        
        fig = px.pie(
            training_counts, 
            values='Count', 
            names='Training Type',
            title='Distribution of Training Types',
            hole=0.4,
            color='Training Type',
            color_discrete_map=color_map,
            custom_data=['Training Type', 'Count']
        )
        fig.update_traces(
            hovertemplate=pie_tooltip
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Training type over time
        type_time_df = training_df.groupby([training_df['Date'].dt.strftime('%Y-%m'), 'training_type']).size().reset_index()
        type_time_df.columns = ['Month', 'Training Type', 'Count']
        type_time_df.sort_values('Count', inplace=True, ascending=False)
        
        fig = px.bar(
            type_time_df,
            x='Month',
            y='Count',
            color='Training Type',
            title='Training Types by Month',
            barmode='stack',
            color_discrete_map=color_map,
            custom_data=['Training Type', 'Count', 'Month']
        )
        fig.update_traces(
            hovertemplate='<span style="color: grey; font-size: 12px"><br>%{customdata[0]}</span><br><span style="color: black; font-size: 15px;">%{customdata[1]} sessions</span><extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Training frequency by type
    st.markdown('<div class="sub-header">Type-specific Insights</div>', unsafe_allow_html=True)
    
    # Get unique training types
    training_types = training_df['training_type'].unique()
    
    if len(training_types) > 0:
        # Create color map from the dataframe
        color_map = dict(zip(training_df['training_type'], training_df['type_color']))
        
        # Create a colored title for each type option
        type_options = [f"<span style='color: {color_map[t]}'>{t}</span>" for t in training_types]
        selected_index = st.selectbox(
            "Select a specific training type to analyze:",
            range(len(type_options)),
            format_func=lambda x: training_types[x]
        )
        selected_type = training_types[selected_index]
        
        # Filter data for the selected type
        type_df = training_df[training_df['training_type'] == selected_type]
        selected_color = color_map[selected_type]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Days between selected training type
            if len(type_df) > 1:
                type_df_sorted = type_df.sort_values('Date')
                days_between = (type_df_sorted['Date'].shift(-1) - type_df_sorted['Date']).dt.days
                days_between = days_between.dropna()
                
                # Create histogram data with bin information
                hist_data = np.histogram(days_between, bins=10)
                bin_counts = hist_data[0]
                bin_edges = hist_data[1]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bin_centers,
                    y=bin_counts,
                    marker_color=selected_color,
                    customdata=list(zip(bin_edges[:-1], bin_edges[1:], bin_counts)),
                    hovertemplate='<span style="color: grey; font-size: 12px">%{customdata[0]:.0f} - %{customdata[1]:.0f} days</span><br><span style="color: black; font-size: 15px;">%{customdata[2]} occurrences</span><extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'Days Between {selected_type} Sessions',
                    xaxis_title='Days Between Sessions',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info(f"Not enough {selected_type} sessions to analyze intervals.")
        
        with col2:
            # Weekday distribution for selected type
            weekday_counts = type_df['weekday'].value_counts()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekday_counts.index,
                y=weekday_counts.values,
                marker_color=selected_color,
                customdata=list(zip(weekday_counts.index, weekday_counts.values)),
                hovertemplate='<span style="color: grey; font-size: 12px">%{customdata[0]}</span><br><span style="color: black; font-size: 15px;">%{customdata[1]} sessions</span><extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Weekday Distribution for {selected_type}',
                xaxis_title='Day of Week',
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def display_weekly_patterns(df):
    """Display weekly training patterns."""
    st.markdown('<div class="sub-header">Weekly Training Patterns</div>', unsafe_allow_html=True)
    
    # Create color map from the dataframe
    color_map = dict(zip(df['training_type'], df['type_color']))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training by day of week
        weekday_counts = df.groupby('weekday')['Trained (1/0)'].agg(['sum', 'count'])
        weekday_counts['percentage'] = (weekday_counts['sum'] / weekday_counts['count']) * 100
        
        # Reorder days of week
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = weekday_counts.reindex(weekday_order)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekday_counts.index,
            y=weekday_counts['percentage'],
            marker_color='#1E88E5',
            customdata=list(zip(weekday_counts.index, weekday_counts['sum'], weekday_counts['count'], weekday_counts['percentage'])),
            hovertemplate='<span style="color: grey; font-size: 12px">%{customdata[0]}</span><br><span style="color: black; font-size: 15px;">%{customdata[3]:.1f}%</span><extra></extra>'
        ))
        
        fig.update_layout(
            title='Training Percentage by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Training %',
            height=400,
            yaxis=dict(
                ticksuffix='%'  # Add % symbol to y-axis labels
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Training types by day of week
        day_type_df = df[df['Trained (1/0)'] == 1].groupby(['weekday', 'training_type']).size().reset_index()
        day_type_df.columns = ['Day of Week', 'Training Type', 'Count']
        
        # Reorder days of week
        day_type_df['Day_Order'] = day_type_df['Day of Week'].apply(lambda x: weekday_order.index(x) if x in weekday_order else 999)
        day_type_df = day_type_df.sort_values('Day_Order')
        day_type_df = day_type_df.drop('Day_Order', axis=1)
        
        fig = px.bar(
            day_type_df,
            x='Day of Week',
            y='Count',
            color='Training Type',
            title='Training Types by Day of Week',
            barmode='stack',
            color_discrete_map=color_map,
            custom_data=['Day of Week', 'Training Type', 'Count']
        )
        fig.update_traces(
            hovertemplate='<span style="color: grey; font-size: 12px">%{customdata[0]}<br>%{customdata[1]}</span><br><span style="color: black; font-size: 15px;">%{customdata[2]} sessions</span><extra></extra>'
        )
        fig.update_layout(
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Rest day analysis
    st.markdown('<div class="sub-header">Rest Day Analysis</div>', unsafe_allow_html=True)
    
    rest_days = df[df['Trained (1/0)'] == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rest days by day of week
        rest_weekday_counts = rest_days['weekday'].value_counts().reindex(weekday_order, fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rest_weekday_counts.index,
            y=rest_weekday_counts.values,
            marker_color='#757575',  # Gray color for rest days
            customdata=list(zip(rest_weekday_counts.index, rest_weekday_counts.values)),
            hovertemplate='<span style="color: grey; font-size: 12px">%{customdata[0]}</span><br><span style="color: black; font-size: 15px;">%{customdata[1]} rest days</span><extra></extra>'
        ))
        
        fig.update_layout(
            title='Rest Days by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Training before/after rest days
        if not rest_days.empty:
            # Get the day before each rest day
            rest_days_sorted = rest_days.sort_values('Date')
            day_before_rest = [date - timedelta(days=1) for date in rest_days_sorted['Date']]
            training_before_rest = []
            
            for date in day_before_rest:
                match = df[df['Date'] == date]
                if not match.empty:
                    if match['Trained (1/0)'].values[0] == 1:
                        training_before_rest.append(match['training_type'].values[0])
                    else:
                        training_before_rest.append('Rest')
                else:
                    training_before_rest.append('Unknown')
            
            # Count training types before rest days
            training_before_counts = pd.Series(training_before_rest).value_counts()
            
            fig = px.pie(
                values=training_before_counts.values,
                names=training_before_counts.index,
                title='Training Before Rest Days',
                hole=0.4,
                custom_data=[training_before_counts.index, training_before_counts.values]
            )
            fig.update_traces(
                hovertemplate='<span style="color: grey; font-size: 12px">%{customdata[0]}</span><br><span style="color: black; font-size: 15px;">%{customdata[1]} times</span><extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No rest days in the selected period.")

def display_raw_data(df):
    """Display the raw data and stats."""
    st.markdown('<div class="sub-header">Raw Data</div>', unsafe_allow_html=True)
    
    # Display basic stats
    st.write("Basic Statistics")
    
    # Convert to pandas before displaying to avoid Arrow conversion issues
    stats_df = df.describe().to_pandas() if hasattr(df, 'to_pandas') else df.describe()
    st.write(stats_df)
    
    # Display raw data with option to download
    st.write("Raw Data")
    st.dataframe(df)
    
    # Add a download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="training_data.csv",
        mime="text/csv"
    )

def main():
    """Main function to run the app."""
    try:
        # Load data
        df = load_data()
        
        # Create sidebar with app info
        with st.sidebar:
            st.markdown("# 🏋️‍♂️")
            st.title("Training Tracker")
            st.info("""
            This dashboard helps you analyze your training patterns and progress.
            
            **Features:**
            - Overview of training consistency
            - Analysis by training type
            - Weekly training patterns
            - Raw data access
            """)
            
            st.markdown("---")
            st.markdown("### Training Goals")
            
            # Create a button to toggle the slider visibility
            if 'show_slider' not in st.session_state:
                st.session_state.show_slider = False
                
            if 'target_days' not in st.session_state:
                st.session_state.target_days = 3
            
            # Display current goal as a bullet point
            st.markdown(f"""
            <span style='color: #1E88E5;'>•</span> <span style='color: #1E88E5; font-weight: bold;'>1 session every {st.session_state.target_days} days</span>
            """, unsafe_allow_html=True)
            
            # Edit button and slider
            if st.button("Edit", key="edit_goal"):
                st.session_state.show_slider = not st.session_state.show_slider
                
            if st.session_state.show_slider:
                new_target = st.slider(
                    "Days between sessions",
                    min_value=1,
                    max_value=7,
                    value=st.session_state.target_days,
                    key="target_slider"
                )
                if new_target != st.session_state.target_days:
                    st.session_state.target_days = new_target
                    st.rerun()
                target_ratio = 1/st.session_state.target_days
            else:
                target_ratio = 1/st.session_state.target_days
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("Built with Streamlit and Python")
            
        # Create the main dashboard
        create_dashboard(df, target_ratio)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please make sure the data source is accessible and properly formatted.")

if __name__ == "__main__":
    main()