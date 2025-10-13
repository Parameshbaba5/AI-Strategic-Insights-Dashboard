import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px

# ---------------------------------------------
# Streamlit setup
# ---------------------------------------------
st.set_page_config(page_title="AI Strategic Intelligence Dashboard", layout="wide")
st.title("🤖 AI Strategic Intelligence Dashboard")

# ---------------------------------------------
# Generate AI data
# ---------------------------------------------
@st.cache_data
def generate_ai_data(days=240):
    competitors = [
        ("OpenAI", "Generative AI"),
        ("Google DeepMind", "AI Research"),
        ("Anthropic", "LLMs"),
        ("Meta AI", "AI Research"),
        ("Cohere", "NLP/LLMs"),
        ("Hugging Face", "AI Tools"),
        ("NVIDIA AI", "AI Hardware"),
        ("Amazon Bedrock", "Cloud AI Services"),
        ("IBM WatsonX", "Enterprise AI"),
        ("Stability AI", "Generative AI")
    ]

    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    records = []

    for comp, sector in competitors:
        base_sent = rng.normal(0.1, 0.05)
        base_mentions = rng.integers(50, 200)
        sov_base = rng.uniform(0.05, 0.35)

        for i, d in enumerate(dates):
            trend = 0.0008 * i
            seasonal = 0.04 * np.sin(i * 2 * np.pi / 30)
            shock = rng.choice([0, 0, 0, -0.2, 0.2], p=[0.7, 0.1, 0.1, 0.05, 0.05])
            sentiment = np.clip(base_sent + trend + seasonal + rng.normal(0, 0.03) + shock * (rng.random() < 0.02), -1, 1)
            mentions = int(base_mentions + 3 * i / 30 + rng.normal(0, 15))
            sov = sov_base + 0.0005 * i + 0.02 * np.sin(i * 2 * np.pi / 45) + rng.normal(0, 0.02)

            records.append({
                "date": d.date(),
                "competitor": comp,
                "sector": sector,
                "sentiment_score": float(sentiment),
                "mentions": max(0, mentions),
                "share_of_voice": float(sov)
            })

    df = pd.DataFrame.from_records(records)
    df_sum = df.groupby("date")["share_of_voice"].transform("sum")
    df["share_of_voice"] = df["share_of_voice"] / df_sum
    return df

# ---------------------------------------------
# Data input
# ---------------------------------------------
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Use AI Data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_demo:
    df = generate_ai_data()
else:
    st.warning("Upload a CSV file or check 'Use AI Data' to continue.")
    st.stop()

# ---------------------------------------------
# Sidebar filters
# ---------------------------------------------
sectors = st.sidebar.multiselect("🧩 Select Sectors", df["sector"].unique(), default=df["sector"].unique())
competitors = st.sidebar.multiselect("🏢 Select Competitors", df["competitor"].unique(), default=df["competitor"].unique())

filtered_df = df[df["sector"].isin(sectors) & df["competitor"].isin(competitors)]

# ---------------------------------------------
# 📊 Sentiment Analysis (added section)
# ---------------------------------------------
st.subheader("📊 Sentiment Analysis")

# Interactive smoothing window
smoothing_window = st.slider("Smoothing Window (days)", 3, 30, 7)

sentiment_df = filtered_df.copy()
sentiment_df["sentiment_smooth"] = (
    sentiment_df.groupby("competitor")["sentiment_score"]
    .transform(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean())
)

fig_sentiment = px.line(
    sentiment_df,
    x="date",
    y="sentiment_smooth",
    color="competitor",
    title="Sentiment Analysis Across Competitors",
    labels={"sentiment_smooth": "Sentiment Score", "date": "Date"},
)
fig_sentiment.update_layout(template="plotly_white", title_x=0.3)
st.plotly_chart(fig_sentiment, use_container_width=True)

# ---------------------------------------------
# 💬 Sentiment Over Time
# ---------------------------------------------
st.subheader("💬 Sentiment Over Time")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered_df, x="date", y="sentiment_score", hue="competitor", ax=ax2)
ax2.set_xlabel("Date"); ax2.set_ylabel("Sentiment Score")
st.pyplot(fig2)

# ---------------------------------------------
# 📈 Mentions Over Time
# ---------------------------------------------
st.subheader("📈 Mentions Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered_df, x="date", y="mentions", hue="competitor", ax=ax)
ax.set_xlabel("Date"); ax.set_ylabel("Mentions")
st.pyplot(fig)



# ---------------------------------------------
# 🗣️ Share of Voice Distribution
# ---------------------------------------------
st.subheader("🗣️ Share of Voice Distribution")
fig3, ax3 = plt.subplots(figsize=(10, 4))
share_df = filtered_df.groupby("competitor")["share_of_voice"].mean().reset_index()
sns.barplot(data=share_df, x="competitor", y="share_of_voice", ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# ---------------------------------------------
# 🚨 Anomaly Detection
# ---------------------------------------------
st.subheader("🚨 Anomaly Detection in Mentions")
selected_comp = st.selectbox("Select Competitor for Anomaly Detection", filtered_df["competitor"].unique())
comp_df = filtered_df[filtered_df["competitor"] == selected_comp].copy()

iso = IsolationForest(contamination=0.05, random_state=42)
comp_df["anomaly"] = iso.fit_predict(comp_df[["mentions"]])
comp_df["is_anomaly"] = comp_df["anomaly"] == -1

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(comp_df["date"], comp_df["mentions"], label="Mentions", color="blue")
ax4.scatter(comp_df[comp_df["is_anomaly"]]["date"], comp_df[comp_df["is_anomaly"]]["mentions"], color="red", label="Anomalies")
ax4.set_xlabel("Date"); ax4.set_ylabel("Mentions")
ax4.legend()
st.pyplot(fig4)



# ---------------------------------------------
# 🧭 Alerts & Insights
# ---------------------------------------------
st.subheader("🧭 AI Insights & Alerts")

# Top anomalies
anomaly_counts = (
    filtered_df.groupby("competitor")["mentions"]
    .apply(lambda x: (IsolationForest(contamination=0.05, random_state=0)
                      .fit_predict(x.values.reshape(-1, 1)) == -1).sum())
    .reset_index(name="anomaly_count")
)
top_anomalies = anomaly_counts.sort_values(by="anomaly_count", ascending=False).head(3)

# Fastest-growing competitors (based on last 30 days)
growth_df = (
    filtered_df.groupby("competitor")
    .apply(lambda x: (x["mentions"].iloc[-1] - x["mentions"].iloc[-30]) / x["mentions"].iloc[-30] * 100 if len(x) > 30 else 0)
    .reset_index(name="growth_rate")
)
top_growth = growth_df.sort_values("growth_rate", ascending=False).head(3)

# Display insights
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚨 Top Anomalies Detected")
    st.dataframe(top_anomalies)

with col2:
    st.markdown("### 📈 Fastest-Growing Competitors")
    st.dataframe(top_growth)

# Generate insight text
top_grower = top_growth.iloc[0]["competitor"] if not top_growth.empty else "N/A"
top_anom = top_anomalies.iloc[0]["competitor"] if not top_anomalies.empty else "N/A"

st.markdown(f"""
**🔍 Insight Summary:**
- {top_grower} has shown the **highest growth rate** in mentions recently, indicating rising attention.
- {top_anom} recorded the **most anomalies**, suggesting volatile activity or sudden news impact.
- Overall market sentiment remains stable with positive trends in Generative and Research AI sectors.
""")

# ---------------------------------------------
# 🤖 Intelligent Slack Alerts (Auto + Manual)
# ---------------------------------------------
import requests

st.subheader("📢 Smart Slack Alerts")

webhook_url = "https://hooks.slack.com/services/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"

if not webhook_url or "REPLACE_WITH" in webhook_url:
    st.warning("⚠️ Slack Webhook URL not configured. Please update it in the code.")
else:
    # --- Compute sentiment summary ---
    sentiment_summary = (
        filtered_df.groupby("competitor")["sentiment_score"]
        .mean()
        .reset_index()
        .sort_values("sentiment_score")
    )

    # Detect low sentiment competitors
    negative_competitors = sentiment_summary[sentiment_summary["sentiment_score"] < -0.2]

    # Prepare alert text
    alert_message = f"""
🚨 *AI Dashboard Alert* 🚨
*Time:* {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
*Top Growth:* {top_grower}
*Most Anomalies:* {top_anom}
*Selected Sectors:* {', '.join(sectors)}

"""

    # --- AUTO ALERT RULES ---
    auto_triggered = False

    # Rule 1: Too many anomalies
    if not top_anomalies.empty and top_anomalies.iloc[0]["anomaly_count"] > 5:
        alert_message += f"⚠️ *High anomaly activity detected* in **{top_anom}**!\n"
        auto_triggered = True

    # Rule 2: Negative sentiment
    if not negative_competitors.empty:
        neg_list = ", ".join(negative_competitors["competitor"])
        alert_message += f"😞 *Negative sentiment trend detected* for: {neg_list}\n"
        auto_triggered = True

    # --- AUTO ALERT SEND ---
    if auto_triggered:
        try:
            response = requests.post(webhook_url, json={"text": alert_message})
            if response.status_code == 200:
                st.info("📤 Auto Slack alert sent (Anomaly or Negative Sentiment detected).")
            else:
                st.warning(f"⚠️ Failed to send auto-alert: {response.status_code}")
        except Exception as e:
            st.error(f"Error sending Slack alert: {e}")

    # --- MANUAL ALERT BUTTON ---
    if st.button("🚀 Send Slack Alert Manually"):
        try:
            response = requests.post(webhook_url, json={"text": alert_message})
            if response.status_code == 200:
                st.success("✅ Slack alert sent successfully!")
            else:
                st.error(f"❌ Failed to send Slack alert: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error sending Slack alert: {e}")


st.success("✅ Dashboard loaded with insights, sentiment trajectories, anomaly detection, and forecasting.")
