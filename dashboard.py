import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="📦 Store Dashboard", layout="wide")

st.title("📊 Store Data Dashboard")

# --- Sidebar Upload Section ---
st.sidebar.header("Upload Your Data")

inventory_file = st.sidebar.file_uploader("Upload Inventory File (CSV/XLSX)", type=["csv", "xlsx"])
sales_file = st.sidebar.file_uploader("Upload Sales File (CSV/XLSX)", type=["csv", "xlsx"])
shipment_file = st.sidebar.file_uploader("Upload Shipment File (CSV/XLSX)", type=["csv", "xlsx"])

# --- Helper function to load data ---
def load_file(file):
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
    return None

# Load each file
inventory_df = load_file(inventory_file)
sales_df = load_file(sales_file)
shipment_df = load_file(shipment_file)

# --- Show uploaded data tables ---
if inventory_df is not None:
    st.subheader("📦 Inventory Data")
    st.dataframe(inventory_df, use_container_width=True)

if sales_df is not None:
    st.subheader("💰 Sales Data")
    st.dataframe(sales_df, use_container_width=True)

if shipment_df is not None:
    st.subheader("🚚 Shipment Data")
    st.dataframe(shipment_df, use_container_width=True)

st.divider()

# --- Inventory Visualization ---
if inventory_df is not None and "Product" in inventory_df.columns and "Quantity" in inventory_df.columns:
    st.subheader("Inventory Overview")
    fig = px.bar(
        inventory_df,
        x="Product",
        y="Quantity",
        color="Quantity",
        title="Inventory Levels by Product"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sales Visualization ---
if sales_df is not None:
    if "Date" in sales_df.columns and "Revenue" in sales_df.columns:
        st.subheader("Revenue Over Time")
        sales_df["Date"] = pd.to_datetime(sales_df["Date"], errors="coerce")
        revenue_by_day = sales_df.groupby("Date")["Revenue"].sum().reset_index()
        fig2 = px.line(revenue_by_day, x="Date", y="Revenue", title="Revenue Over Time")
        st.plotly_chart(fig2, use_container_width=True)

    if "Product" in sales_df.columns and "Revenue" in sales_df.columns:
        st.subheader("Revenue by Product")
        revenue_by_product = sales_df.groupby("Product")["Revenue"].sum().reset_index()
        fig3 = px.bar(
            revenue_by_product,
            x="Product",
            y="Revenue",
            color="Revenue",
            title="Total Revenue per Product"
        )
        st.plotly_chart(fig3, use_container_width=True)

# --- Shipment Visualization ---
if shipment_df is not None:
    st.subheader("Shipment Overview")

    # Convert Date column if it exists
    if "Date" in shipment_df.columns:
        shipment_df["Date"] = pd.to_datetime(shipment_df["Date"], errors="coerce")

    # Shipments over time
    if "Date" in shipment_df.columns:
        shipments_by_date = shipment_df.groupby("Date").size().reset_index(name="Shipments")
        fig4 = px.line(shipments_by_date, x="Date", y="Shipments", title="Shipments Over Time")
        st.plotly_chart(fig4, use_container_width=True)

    # Shipments by product
    if "Product" in shipment_df.columns:
        shipments_by_product = shipment_df.groupby("Product").size().reset_index(name="Shipments")
        fig5 = px.bar(
            shipments_by_product,
            x="Product",
            y="Shipments",
            color="Shipments",
            title="Shipments by Product"
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Shipments by supplier (if available)
    if "Supplier" in shipment_df.columns:
        shipments_by_supplier = shipment_df.groupby("Supplier").size().reset_index(name="Shipments")
        fig6 = px.pie(
            shipments_by_supplier,
            names="Supplier",
            values="Shipments",
            title="Shipments by Supplier"
        )
        st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.caption("📈 Built with Streamlit + Plotly | Drop in your store data (Inventory, Sales, Shipments)")
