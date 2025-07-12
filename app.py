import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt

st.set_page_config(page_title="Customer Segmentation", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("data/cust segmentation.xlsx")
    df.columns = df.columns.str.lower()
    return df

df = load_data()

# Group by customer
df_grouped = df.groupby('customer_id').agg({
    'recency': 'first',
    'frequency': 'first',
    'monetary': 'first',
    'sales': 'sum',
    'quantity': 'sum',
    'profit': 'sum',
    'segment': 'first',
    'city': 'first',
    'ship_mode': 'first',
    'category': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'subcategory': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
}).reset_index()

# Clustering based on grouped data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df_grouped[['recency', 'frequency', 'monetary']])
model = KMeans(n_clusters=3, random_state=42).fit(rfm_scaled)

# Re-label clusters based on logic (recency low + monetary high → best customers)
initial_clusters = model.predict(rfm_scaled)
centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=['recency', 'frequency', 'monetary'])
sorted_clusters = centroids.sort_values(['recency', 'monetary'], ascending=[True, False]).index.tolist()
label_map = {
    sorted_clusters[0]: 1,  # Best
    sorted_clusters[1]: 2,  # Potential
    sorted_clusters[2]: 0   # Inactive
}
df_grouped['cluster'] = pd.Series(initial_clusters).map(label_map)

# Sidebar page selector
st.sidebar.header("🎛️ Navigation")
selected_page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🧪 Manual Prediction"])

if selected_page == "📊 Dashboard":
    selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(df_grouped['cluster'].unique()))
    show_raw = st.sidebar.checkbox("📂 Show Raw Data")

    # Customer distribution pie chart
    cluster_counts = df_grouped['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    chart = alt.Chart(cluster_counts).mark_arc(innerRadius=50).encode(
        theta='count',
        color='cluster:N',
        tooltip=['cluster', 'count']
    ).properties(
        title=alt.TitleParams(text='CUSTOMER DISTRIBUTION BY CLUSTER', fontSize=16, fontWeight='bold', anchor='middle')
    )
    st.sidebar.altair_chart(chart, use_container_width=True)

    filtered_df = df_grouped[df_grouped['cluster'] == selected_cluster]

    st.title("Customer Segmentation")
    st.markdown("Customer segmentation based on **Recency**, **Frequency**, and **Monetary** (RFM)")

    if show_raw:
        st.subheader("📄 Raw Customer Data")
        st.dataframe(filtered_df)

    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Descriptive Statistics", "📌 Cluster Highlights"])

    with tab1:
        st.subheader(f"📊 Average RFM - Cluster {selected_cluster}")
        cluster_data = filtered_df[['recency', 'frequency', 'monetary']]

        r_mean = cluster_data['recency'].mean()
        f_mean = cluster_data['frequency'].mean()
        m_mean = cluster_data['monetary'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<h2 style='text-align:center'>{r_mean:.2f}</h2>", unsafe_allow_html=True)
            r_df = pd.DataFrame({'Metric': ['Recency'], 'Value': [r_mean]})
            r_chart = alt.Chart(r_df).mark_bar(color='#FF4C4C').encode(
                x=alt.X('Metric', sort=None),
                y='Value',
                tooltip=['Metric', 'Value']
            ).interactive().properties(height=300)
            st.altair_chart(r_chart, use_container_width=True)

        with col2:
            st.markdown(f"<h2 style='text-align:center'>{f_mean:.2f}</h2>", unsafe_allow_html=True)
            f_df = pd.DataFrame({'Metric': ['Frequency'], 'Value': [f_mean]})
            f_chart = alt.Chart(f_df).mark_bar(color='#3B9CFF').encode(
                x=alt.X('Metric', sort=None),
                y='Value',
                tooltip=['Metric', 'Value']
            ).interactive().properties(height=300)
            st.altair_chart(f_chart, use_container_width=True)

        with col3:
            st.markdown(f"<h2 style='text-align:center'>{m_mean:,.2f}</h2>", unsafe_allow_html=True)
            m_df = pd.DataFrame({'Metric': ['Monetary'], 'Value': [m_mean]})
            m_chart = alt.Chart(m_df).mark_bar(color='#00C49F').encode(
                x=alt.X('Metric', sort=None),
                y='Value',
                tooltip=['Metric', 'Value']
            ).interactive().properties(height=300)
            st.altair_chart(m_chart, use_container_width=True)

        st.markdown("### 👤 Customer Profile")
        if selected_cluster == 1:
            st.success("""
### 🏆 Best Customers
Low **Recency**, high **Monetary** → active, high-value, and frequent buyers.
""")
        elif selected_cluster == 0:
            st.warning("""
### ⚠️ Inactive Customers
High **Recency**, low **Monetary** → haven’t purchased in a while, low value. High churn risk.
""")
        elif selected_cluster == 2:
            st.info("""
### 🚀 Potential Customers
Low **Recency**, low **Monetary** → active but low spending. Likely new or potential customers.
""")

        st.markdown("### 📊 Total Customers")
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 10px; background-color: #f0f2f6; text-align: center;'>
            <h3 style='margin: 0;'>👥 {len(filtered_df)} customers</h3>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader(f"📈 Descriptive Statistics - Cluster {selected_cluster}")
        st.dataframe(filtered_df[['sales', 'quantity', 'profit']].describe())

    with tab3:
        st.subheader(f"📌 Cluster Highlights - Cluster {selected_cluster}")

        def chart_title(title):
            return alt.TitleParams(text=title.upper(), fontSize=16, fontWeight='bold', anchor='middle')

        col1, col2 = st.columns(2)
        with col1:
            seg_chart = alt.Chart(filtered_df).mark_arc(innerRadius=60).encode(
                theta='count():Q',
                color=alt.Color('segment:N', scale=alt.Scale(scheme='dark2')),
                tooltip=['segment', 'count()']
            ).properties(title=chart_title("Main Segment")).interactive()
            st.altair_chart(seg_chart, use_container_width=True)

        with col2:
            city_df = filtered_df['city'].value_counts().reset_index()
            city_df.columns = ['city', 'count']
            top5_city_chart = alt.Chart(city_df.head(5)).mark_bar(color="#FF914D").encode(
                x=alt.X('count:Q', title='Number of Customers'),
                y=alt.Y('city:N', sort='-x'),
                tooltip=['city', 'count']
            ).properties(title=chart_title("Top 5 Cities by Customers")).interactive()
            st.altair_chart(top5_city_chart, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            mode_df = filtered_df['ship_mode'].value_counts().reset_index()
            mode_df.columns = ['ship_mode', 'count']
            ship_chart = alt.Chart(mode_df).mark_bar().encode(
                x=alt.X('ship_mode', sort='-y'),
                y='count',
                color=alt.value('#FFD700'),
                tooltip=['ship_mode', 'count']
            ).properties(title=chart_title("Shipping Mode")).interactive()
            st.altair_chart(ship_chart, use_container_width=True)

        with col4:
            cat_chart = alt.Chart(filtered_df).mark_arc().encode(
                theta='count():Q',
                color=alt.Color('category:N', scale=alt.Scale(scheme='tableau20')),
                tooltip=['category', 'count()']
            ).properties(title=chart_title("Top Category")).interactive()
            st.altair_chart(cat_chart, use_container_width=True)

        sub_df = filtered_df[filtered_df['category'] == 'Office Supplies']
        sub_chart = alt.Chart(sub_df).mark_arc(innerRadius=50).encode(
            theta='count():Q',
            color=alt.Color('subcategory:N', scale=alt.Scale(scheme='set3')),
            tooltip=['subcategory', 'count()']
        ).properties(title=chart_title("Top Office Supplies Subcategories")).interactive()
        st.altair_chart(sub_chart, use_container_width=True)

elif selected_page == "🧪 Manual Prediction":
    st.title("🧪 Manual Cluster Prediction")
    st.markdown("Simulate your own customer RFM score and predict which cluster it would fall into.")

    r = st.slider("Recency", int(df_grouped['recency'].min()), int(df_grouped['recency'].max()), 30)
    f = st.slider("Frequency", int(df_grouped['frequency'].min()), int(df_grouped['frequency'].max()), 5)
    m = st.slider("Monetary", int(df_grouped['monetary'].min()), int(df_grouped['monetary'].max()), 500)

    if st.button("Predict Cluster"):
        rfm_input = pd.DataFrame([[r, f, m]], columns=['recency', 'frequency', 'monetary'])
        rfm_scaled_input = scaler.transform(rfm_input)
        raw_pred = model.predict(rfm_scaled_input)[0]
        pred_cluster = label_map.get(raw_pred, raw_pred)

        label_text = {
            1: "🏆 Best Customers",
            0: "⚠️ Inactive Customers",
            2: "🚀 Potential Customers"
        }
        st.success(f"Predicted Cluster: {pred_cluster} → {label_text[pred_cluster]}")













