import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.runtime
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set UTF-8 encoding for console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Indian Retail Customer Segmentation",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    @st.cache_data
    def load_and_segment_data():
        """Load raw sales data, run data cleaning and K-Means segmentation pipeline."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        candidate_inputs = [
            os.path.join(script_dir, "Indian Retail Store.csv"),
            os.path.join(script_dir, "data", "indian_retail_sales.csv"),
            "Indian Retail Store.csv"
        ]
        
        input_csv = None
        for candidate in candidate_inputs:
            if os.path.exists(candidate):
                input_csv = candidate
                break
                
        if not input_csv:
            st.error("⚠️ Raw sales dataset `Indian Retail Store.csv` not found!")
            st.stop()
            
        # 1. Load Data
        df = pd.read_csv(input_csv)
        
        # 2. Clean Data
        df['visit_date'] = pd.to_datetime(df['visit_date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['customer_name', 'total_amount', 'visit_date']).copy()
        df = df.drop_duplicates()
        
        # 3. Customer Level Aggregation (RFM Metrics)
        max_date = df['visit_date'].max()
        snapshot_date = max_date + pd.Timedelta(days=1)

        def get_mode(series):
            modes = series.mode()
            return modes.iloc[0] if not modes.empty else series.iloc[0]

        customer_df = df.groupby('customer_name').agg(
            City=('city', 'first'),
            Total_Spend=('total_amount', 'sum'),
            Frequency=('bill_id', 'nunique'),
            Last_Visit=('visit_date', 'max'),
            Total_Quantity=('quantity', 'sum'),
            Favorite_Category=('product_category', get_mode),
            Favorite_Payment=('payment_method', get_mode),
            Store_Type=('store_type', get_mode)
        ).reset_index()

        customer_df['Recency_Days'] = (snapshot_date - customer_df['Last_Visit']).dt.days
        customer_df['Avg_Transaction_Value'] = (customer_df['Total_Spend'] / customer_df['Frequency']).round(2)

        # 4. K-Means Clustering
        feature_cols = ['Total_Spend', 'Frequency', 'Recency_Days']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(customer_df[feature_cols])

        n_clusters = min(3, len(customer_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer_df['Cluster'] = kmeans.fit_predict(scaled_features)

        # Human-readable Segment Names based on Total Spend
        cluster_spend_means = customer_df.groupby('Cluster')['Total_Spend'].mean().sort_values(ascending=False)
        segment_labels = {
            cluster_spend_means.index[0]: 'High Value Spenders',
            cluster_spend_means.index[1]: 'Regular Shoppers' if len(cluster_spend_means) > 1 else 'Shoppers',
            cluster_spend_means.index[2]: 'Budget Buyers' if len(cluster_spend_means) > 2 else 'Low Spenders'
        }
        customer_df['Segment_Name'] = customer_df['Cluster'].map(segment_labels)
        
        return customer_df

    # Custom CSS for Premium Design Aesthetics
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #e0e6ed;
        }
        .header-container {
            background: linear-gradient(135deg, #1e2640 0%, #0d1b2a 100%);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #2e3a59;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .header-title {
            color: #4cc9f0;
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .header-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            margin-top: 6px;
        }
        .metric-card {
            background: linear-gradient(145deg, #1b2234 0%, #151b28 100%);
            border: 1px solid #2a354f;
            border-radius: 10px;
            padding: 18px 20px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .metric-label {
            font-size: 0.88rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 6px;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            color: #38bdf8;
            margin-top: 4px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #2a354f;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    df = load_and_segment_data()

    # Header Banner
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🛍️ Indian Retail Customer Segmentation Dashboard</div>
        <div class="header-subtitle">Behavioral Clustering Analytics (RFM & K-Means)</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Filters
    st.sidebar.header("🔍 Interactive Filters")
    st.sidebar.markdown("Filter segment insights and customer records:")

    all_segments = list(df['Segment_Name'].unique())
    selected_segments = st.sidebar.multiselect("Select Customer Segments", options=all_segments, default=all_segments)

    all_categories = list(df['Favorite_Category'].unique())
    selected_categories = st.sidebar.multiselect("Select Favorite Category", options=all_categories, default=all_categories)

    all_payments = list(df['Favorite_Payment'].dropna().unique())
    selected_payments = st.sidebar.multiselect("Select Payment Method", options=all_payments, default=all_payments)

    search_term = st.sidebar.text_input("🔎 Search Customer Name", "")

    # Apply Filters
    filtered_df = df[
        (df['Segment_Name'].isin(selected_segments)) &
        (df['Favorite_Category'].isin(selected_categories)) &
        (df['Favorite_Payment'].isin(selected_payments))
    ]

    if search_term:
        filtered_df = filtered_df[filtered_df['customer_name'].str.contains(search_term, case=False, na=False)]

    # KPI Section
    st.subheader("📌 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(filtered_df)
    total_revenue = filtered_df['Total_Spend'].sum()
    avg_spend = filtered_df['Total_Spend'].mean() if total_customers > 0 else 0
    top_segment = filtered_df.groupby('Segment_Name')['Total_Spend'].sum().idxmax() if total_customers > 0 else "N/A"

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-subtitle">Filtered Scope</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">₹{total_revenue:,.0f}</div>
            <div class="metric-subtitle">Gross Spend</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Customer Spend</div>
            <div class="metric-value">₹{avg_spend:,.0f}</div>
            <div class="metric-subtitle">Per Customer</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Segment (Revenue)</div>
            <div class="metric-value" style="font-size: 1.35rem; color: #4cc9f0;">{top_segment}</div>
            <div class="metric-subtitle">Highest Value Cluster</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Segment Visualizations
    st.subheader("📊 Segment Performance Analytics")
    col_chart1, col_chart2 = st.columns(2)

    segment_colors = {
        'High Value Spenders': '#4cc9f0',
        'Regular Shoppers': '#4895ef',
        'Budget Buyers': '#f72585'
    }

    with col_chart1:
        st.markdown("##### 👥 Customers per Segment")
        customer_counts = filtered_df.groupby('Segment_Name', as_index=False).agg(
            Count=('customer_name', 'count')
        ).sort_values(by='Count', ascending=False)
        
        fig_customers = px.bar(
            customer_counts,
            x='Segment_Name',
            y='Count',
            color='Segment_Name',
            color_discrete_map=segment_colors,
            text='Count',
            labels={'Segment_Name': 'Customer Segment', 'Count': 'Number of Customers'}
        )
        fig_customers.update_traces(texttemplate='%{text}', textposition='outside')
        fig_customers.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#1e293b'),
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_customers, use_container_width=True)

    with col_chart2:
        st.markdown("##### 💰 Revenue per Segment (₹)")
        revenue_sums = filtered_df.groupby('Segment_Name', as_index=False).agg(
            Revenue=('Total_Spend', 'sum')
        ).sort_values(by='Revenue', ascending=False)
        
        fig_revenue = px.bar(
            revenue_sums,
            x='Segment_Name',
            y='Revenue',
            color='Segment_Name',
            color_discrete_map=segment_colors,
            text_auto='.2s',
            labels={'Segment_Name': 'Customer Segment', 'Revenue': 'Total Revenue (₹)'}
        )
        fig_revenue.update_traces(texttemplate='₹%{y:,.0f}', textposition='outside')
        fig_revenue.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#1e293b'),
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    # Additional Deep-Dive Tabs
    tab1, tab2 = st.tabs(["🛒 Category Distribution by Segment", "⚡ Recency & Frequency Matrix"])

    with tab1:
        cat_segment = filtered_df.groupby(['Segment_Name', 'Favorite_Category']).size().reset_index(name='Count')
        fig_cat = px.bar(
            cat_segment,
            x='Segment_Name',
            y='Count',
            color='Favorite_Category',
            title="Product Category Preference Across Segments",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_cat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            yaxis=dict(gridcolor='#1e293b'),
            height=380
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with tab2:
        fig_scatter = px.scatter(
            filtered_df,
            x='Recency_Days',
            y='Total_Spend',
            color='Segment_Name',
            size='Total_Quantity',
            hover_name='customer_name',
            hover_data=['City', 'Favorite_Category', 'Favorite_Payment'],
            color_discrete_map=segment_colors,
            labels={'Recency_Days': 'Recency (Days Ago)', 'Total_Spend': 'Total Spend (₹)'},
            title="Customer Spend vs Recency"
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            height=380
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Filterable Table Section
    st.subheader("📋 Segmented Customers Directory")
    st.markdown("Explore individual customer profiles, segments, spend metrics, and preferences:")

    display_cols = [
        'customer_name', 'City', 'Segment_Name', 'Total_Spend', 
        'Recency_Days', 'Favorite_Category', 'Favorite_Payment', 'Store_Type'
    ]

    st.dataframe(
        filtered_df[display_cols].sort_values(by='Total_Spend', ascending=False),
        column_config={
            "customer_name": "Customer Name",
            "City": "City",
            "Segment_Name": "Segment",
            "Total_Spend": st.column_config.NumberColumn("Total Spend (₹)", format="₹%d"),
            "Recency_Days": st.column_config.NumberColumn("Recency (Days)", format="%d days"),
            "Favorite_Category": "Favorite Category",
            "Favorite_Payment": "Payment Method",
            "Store_Type": "Store Type"
        },
        use_container_width=True,
        hide_index=True
    )

    # Download CSV
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Customers CSV",
        data=csv_data,
        file_name="filtered_segmented_customers.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    if streamlit.runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli
        extra_args = sys.argv[1:]
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + extra_args
        sys.exit(stcli.main())
