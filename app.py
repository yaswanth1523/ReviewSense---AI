
import streamlit as st
import joblib
import pandas as pd
import textwrap
import plotly.express as px

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main application area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #17243B;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    /* Sidebar radio button spacing */
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 5px;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #17243B;
    }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------


@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.joblib")


model = load_model()


# --------------------------------------------------
# 4. SIDEBAR BRANDING AND NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <h2 style="color: white; margin-bottom: 0;">
                🍽️ ReviewSense AI
            </h2>
            <p style="color: #AAB8CC; font-size: 11px;
                      letter-spacing: 1px;">
                RESTAURANT INTELLIGENCE
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("NAVIGATION")

    page_options = [
        "Overview",
        "Sentiment Analyzer",
        "Batch Analysis",
        "Analytics",
        "Model Performance",
        "About the Model"
    ]

    active_page = st.radio(
        "Main Navigation",
        page_options,
        label_visibility="collapsed",
        key="main_navigation"
    )

    st.divider()

    st.markdown(
        """
        <p style="color: #AAB8CC; font-size: 12px;">
            Powered by Machine Learning<br>
            Built with Python & Streamlit
        </p>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 6. PAGE: OVERVIEW
# --------------------------------------------------

if active_page == "Overview":

    # HERO SECTION

    hero_html = """
    <div style="
        background: linear-gradient(120deg, #17243B, #354B72);
        padding: 38px;
        border-radius: 16px;
        color: white;
        margin-bottom: 28px;
    ">

    <p style="
        color: #B9C8E0;
        font-size: 13px;
        letter-spacing: 2px;
        font-weight: 600;
    ">
        AI-POWERED RESTAURANT INTELLIGENCE
    </p>

    <h1 style="
        color: white;
        font-size: 38px;
        margin-bottom: 12px;
    ">
        Understand Every Review.
    </h1>

    <p style="
        color: #E0E8F5;
        font-size: 17px;
        margin-bottom: 0;
    ">
        Transform restaurant feedback into meaningful
        sentiment insights using Natural Language
        Processing and Machine Learning.
    </p>

    </div>
    """

    st.markdown(
        textwrap.dedent(hero_html),
        unsafe_allow_html=True
    )

    # KPI SECTION
    st.subheader("Project at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                label="Dataset Records",
                value="996",
                help="Records remaining after duplicate removal."
            )
            st.caption("Cleaned reviews")

    with col2:
        with st.container(border=True):
            st.metric(
                label="Test Accuracy",
                value="79.50%",
                help="Accuracy measured on the held-out test set."
            )
            st.caption("Model accuracy")

    with col3:
        with st.container(border=True):
            st.metric(
                label="Test F1-Score",
                value="79.80%"
            )
            st.caption("Positive class F1-score")

    with col4:
        with st.container(border=True):
            st.metric(
                label="Algorithm",
                value="BNB"
            )
            st.caption("Naive Bayes classifier")

    st.write("")

    # MACHINE LEARNING WORKFLOW
    st.subheader("How ReviewSense AI Works")

    workflow_col1, workflow_col2, workflow_col3, workflow_col4 = (
        st.columns(4)
    )

    with workflow_col1:
        with st.container(border=True):
            st.markdown("### 📥")
            st.markdown("**01. Review Input**")
            st.caption(
                "Accept restaurant review text from users."
            )

    with workflow_col2:
        with st.container(border=True):
            st.markdown("### 🧹")
            st.markdown("**02. NLP Processing**")
            st.caption(
                "Clean text, expand contractions, remove "
                "stopwords and lemmatize."
            )

    with workflow_col3:
        with st.container(border=True):
            st.markdown("### 🔢")
            st.markdown("**03. Feature Extraction**")
            st.caption(
                "Convert text into binary features using "
                "CountVectorizer."
            )

    with workflow_col4:
        with st.container(border=True):
            st.markdown("### 🧠")
            st.markdown("**04. Prediction**")
            st.caption(
                "Classify reviews using Bernoulli Naive Bayes."
            )

    st.write("")

    # APPLICATION FEATURES
    st.subheader("Explore ReviewSense AI")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        with st.container(border=True):
            st.markdown("### 🔍 Sentiment Analyzer")
            st.write(
                "Analyze individual restaurant reviews "
                "and predict positive or negative sentiment."
            )
            st.caption("Single-review prediction")

        with st.container(border=True):
            st.markdown("### 📊 Sentiment Analytics")
            st.write(
                "Explore sentiment distributions and "
                "insights from analyzed reviews."
            )
            st.caption("Interactive data visualization")

        with st.container(border=True):
            st.markdown("### 📈 Model Performance")
            st.write(
                "Understand the model's accuracy, precision, "
                "recall, F1-score and confusion matrix."
            )
            st.caption("Model evaluation")

    with feature_col2:
        with st.container(border=True):
            st.markdown("### 📂 Batch Analysis")
            st.write(
                "Upload CSV files and analyze multiple "
                "restaurant reviews together."
            )
            st.caption("Bulk sentiment prediction")

        with st.container(border=True):
            st.markdown("### 🧠 About the Model")
            st.write(
                "Explore the dataset, NLP preprocessing, "
                "feature extraction and model architecture."
            )
            st.caption("Machine Learning insights")

        with st.container(border=True):
            st.markdown("### 📥 Export Results")
            st.write(
                "Download analyzed reviews and their "
                "predicted sentiment results."
            )
            st.caption("Downloadable reports")

    # MODEL HIGHLIGHT
    st.write("")
    st.subheader("Model Information")

    with st.container(border=True):

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.markdown("**Feature Extraction**")
            st.write("Binary CountVectorizer")

        with info_col2:
            st.markdown("**Classification Model**")
            st.write("Bernoulli Naive Bayes")

        with info_col3:
            st.markdown("**Best Hyperparameter**")
            st.write("Alpha = 0.5")

    st.caption(
        "ReviewSense AI | Machine Learning-powered "
        "Restaurant Review Sentiment Classification"
    )

# --------------------------------------------------
# 7. PAGE: SENTIMENT ANALYZER
# --------------------------------------------------

elif active_page == "Sentiment Analyzer":

    # Page introduction
    st.title("🔍 Sentiment Analyzer")

    st.write(
        "Discover the sentiment behind restaurant reviews "
        "using our trained Naive Bayes model."
    )

    st.divider()

    # Sample reviews
    sample_reviews = {
        "Positive - Excellent food": (
            "The food was absolutely delicious and "
            "the service was excellent."
        ),
        "Negative - Terrible experience": (
            "I don't like this restaurant. "
            "The food was terrible."
        ),
        "Mixed - Good ambience, slow service": (
            "The ambience was good but the service "
            "was very slow."
        ),
        "Positive - Recommendation": (
            "I would definitely recommend this place."
        )
    }

    # Initialize input state
    if "review_input" not in st.session_state:
        st.session_state.review_input = ""

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    # Review input section
    st.subheader("Write or Select a Restaurant Review")

    selected_sample = st.selectbox(
        "Try an example review",
        options=[
            "Write my own review",
            *sample_reviews.keys()
        ],
        key="sample_review_selector"
    )

    # Load selected example into the text area
    if selected_sample != "Write my own review":
        st.session_state.review_input = (
            sample_reviews[selected_sample]
        )

    review = st.text_area(
        "Your restaurant review",
        key="review_input",
        placeholder=(
            "Example: The food was delicious and "
            "the service was excellent..."
        ),
        height=180
    )

    # Review statistics
    character_count = len(review)
    word_count = len(review.split())

    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        st.metric("Characters", character_count)

    with stat_col2:
        st.metric("Words", word_count)

    with stat_col3:
        st.metric(
            "Input Status",
            "Ready" if review.strip() else "Empty"
        )

    # Prediction button
    predict_button = st.button(
        "✨ Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

    # Execute prediction
    if predict_button:

        if not review.strip():

            st.warning(
                "Please enter a restaurant review before analyzing."
            )

        else:

            review_input = pd.Series([review])

            # Generate prediction
            prediction = model.predict(review_input)[0]

            # Get class probabilities
            probabilities = model.predict_proba(review_input)[0]

            classes = list(model.classes_)

            negative_probability = (
                probabilities[classes.index(0)] * 100
            )

            positive_probability = (
                probabilities[classes.index(1)] * 100
            )

            # Determine predicted sentiment
            sentiment = (
                "Positive" if prediction == 1 else "Negative"
            )

            # Store prediction in session history
            result = {
                "Review": review,
                "Sentiment": sentiment,
                "Negative Probability": round(
                    negative_probability, 2
                ),
                "Positive Probability": round(
                    positive_probability, 2
                )
            }

            st.session_state.analysis_history.insert(0, result)

            # Keep the latest 50 predictions
            st.session_state.analysis_history = (
                st.session_state.analysis_history[:50]
            )

            # Display results
            st.divider()

            st.subheader("Analysis Result")

            if prediction == 1:
                st.success("😊 Positive Sentiment Detected")

            else:
                st.error("😞 Negative Sentiment Detected")

            # Probability visualization
            st.subheader("Estimated Sentiment Probabilities")

            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:

                st.metric(
                    "Negative",
                    f"{negative_probability:.2f}%"
                )

                st.progress(
                    negative_probability / 100
                )

            with probability_col2:

                st.metric(
                    "Positive",
                    f"{positive_probability:.2f}%"
                )

                st.progress(
                    positive_probability / 100
                )

            st.caption(
                "These values are estimated class probabilities "
                "from the Naive Bayes model, not guaranteed "
                "confidence scores."
            )

    # Analysis history
    st.divider()

    st.subheader("Recent Analysis History")

    if st.session_state.analysis_history:

        history_df = pd.DataFrame(
            st.session_state.analysis_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Analysis History"):

            st.session_state.analysis_history = []

            st.rerun()

    else:

        st.info(
            "Your recent review predictions will appear here."
        )


# --------------------------------------------------
# 8. PAGE: BATCH ANALYSIS
# --------------------------------------------------

elif active_page == "Batch Analysis":

    st.title("📂 Batch Sentiment Analysis")

    st.write(
        "Upload a CSV file containing restaurant reviews "
        "and analyze multiple reviews in one operation."
    )

    st.divider()

    # CSV upload
    st.subheader("Upload Your Review Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing restaurant reviews."
    )

    if uploaded_file is not None:

        try:

            # Read uploaded CSV
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.warning(
                    "The uploaded CSV file contains no records."
                )

            elif len(uploaded_df.columns) == 0:
                st.warning(
                    "The uploaded file does not contain columns."
                )

            else:

                st.success("CSV file uploaded successfully!")

                # Dataset preview
                st.subheader("Dataset Preview")

                st.write(
                    f"Rows: {uploaded_df.shape[0]} | "
                    f"Columns: {uploaded_df.shape[1]}"
                )

                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                # Select review column
                st.subheader("Select the Review Column")

                review_column = st.selectbox(
                    "Which column contains restaurant reviews?",
                    options=uploaded_df.columns,
                    key="batch_review_column"
                )

                # Prepare review data
                review_series = uploaded_df[review_column]

                valid_mask = (
                    review_series.notna()
                    & review_series.astype(str).str.strip().ne("")
                )

                valid_reviews = review_series[valid_mask].astype(str)

                total_rows = len(uploaded_df)
                valid_count = len(valid_reviews)
                skipped_count = total_rows - valid_count

                st.write(
                    f"Valid reviews: {valid_count} | "
                    f"Skipped rows: {skipped_count}"
                )

                # Analyze button
                analyze_batch = st.button(
                    "✨ Analyze All Reviews",
                    type="primary",
                    use_container_width=True,
                    disabled=(valid_count == 0)
                )

                if analyze_batch:

                    with st.spinner(
                        "Analyzing restaurant reviews..."
                    ):

                        # Predict all valid reviews together
                        predictions = model.predict(valid_reviews)

                        probabilities = model.predict_proba(
                            valid_reviews
                        )

                        classes = list(model.classes_)

                        negative_index = classes.index(0)
                        positive_index = classes.index(1)

                        negative_probabilities = (
                            probabilities[:, negative_index] * 100
                        )

                        positive_probabilities = (
                            probabilities[:, positive_index] * 100
                        )

                        # Create a copy of the original dataset
                        results_df = uploaded_df.copy()

                        # Add prediction columns
                        results_df["Predicted Sentiment"] = pd.NA
                        results_df["Negative Probability (%)"] = pd.NA
                        results_df["Positive Probability (%)"] = pd.NA

                        # Fill results only for valid reviews
                        results_df.loc[
                            valid_mask, "Predicted Sentiment"
                        ] = [
                            "Positive" if prediction == 1 else "Negative"
                            for prediction in predictions
                        ]

                        results_df.loc[
                            valid_mask, "Negative Probability (%)"
                        ] = negative_probabilities.round(2)

                        results_df.loc[
                            valid_mask, "Positive Probability (%)"
                        ] = positive_probabilities.round(2)

                        # Store results in session state
                        st.session_state.batch_results = results_df

                        st.session_state.batch_summary = {
                            "total": total_rows,
                            "valid": valid_count,
                            "skipped": skipped_count,
                            "positive": int(
                                (predictions == 1).sum()
                            ),
                            "negative": int(
                                (predictions == 0).sum()
                            )
                        }

                # Display results after analysis
                if "batch_results" in st.session_state:

                    results_df = st.session_state.batch_results
                    summary = st.session_state.batch_summary

                    st.divider()

                    st.subheader("Batch Analysis Results")

                    positive_count = summary["positive"]
                    negative_count = summary["negative"]
                    analyzed_count = summary["valid"]

                    positive_percentage = (
                        positive_count / analyzed_count * 100
                    )

                    negative_percentage = (
                        negative_count / analyzed_count * 100
                    )

                    # Summary metrics
                    metric1, metric2, metric3, metric4 = (
                        st.columns(4)
                    )

                    with metric1:
                        st.metric(
                            "Total Rows",
                            summary["total"]
                        )

                    with metric2:
                        st.metric(
                            "Analyzed Reviews",
                            analyzed_count
                        )

                    with metric3:
                        st.metric(
                            "Positive Reviews",
                            positive_count
                        )

                    with metric4:
                        st.metric(
                            "Negative Reviews",
                            negative_count
                        )

                    # Sentiment percentages
                    st.subheader("Sentiment Distribution")

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        st.metric(
                            "Positive",
                            f"{positive_percentage:.2f}%"
                        )
                        st.progress(
                            positive_percentage / 100
                        )

                    with chart_col2:
                        st.metric(
                            "Negative",
                            f"{negative_percentage:.2f}%"
                        )
                        st.progress(
                            negative_percentage / 100
                        )

                    # Results preview
                    st.subheader("Predicted Reviews")

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # CSV download
                    csv_data = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Analysis Results",
                        data=csv_data,
                        file_name="reviewsense_batch_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except pd.errors.EmptyDataError:

            st.error(
                "The uploaded file is empty or does not contain "
                "a valid CSV table."
            )

        except pd.errors.ParserError:

            st.error(
                "Unable to parse this CSV file. "
                "Please check its formatting."
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: {e}"
            )


# --------------------------------------------------
# 9. PAGE: SENTIMENT ANALYTICS
# --------------------------------------------------

elif active_page == "Analytics":

    st.title("📊 Sentiment Analytics")

    st.write(
        "Explore sentiment distributions and probability "
        "insights from your analyzed restaurant reviews."
    )

    st.divider()

    # ----------------------------------------------
    # 1. SELECT ANALYTICS DATA SOURCE
    # ----------------------------------------------

    st.subheader("Choose Analysis Data")

    available_sources = []

    if st.session_state.get("analysis_history"):
        available_sources.append("Single Review History")

    if "batch_results" in st.session_state:
        available_sources.append("Batch Analysis Results")

    if not available_sources:

        st.info(
            "No analysis data is available yet. "
            "Analyze individual reviews or upload a CSV "
            "through the Batch Analysis page first."
        )

    else:

        selected_source = st.radio(
            "Select your data source",
            available_sources,
            horizontal=True
        )

        # ------------------------------------------
        # 2. PREPARE ANALYTICS DATA
        # ------------------------------------------

        if selected_source == "Single Review History":

            analytics_df = pd.DataFrame(
                st.session_state.analysis_history
            )

            analytics_df = analytics_df.rename(
                columns={
                    "Negative Probability": "Negative Probability (%)",
                    "Positive Probability": "Positive Probability (%)"
                }
            )

        else:

            analytics_df = st.session_state.batch_results.copy()

            analytics_df = analytics_df.rename(
                columns={
                    "Predicted Sentiment": "Sentiment"
                }
            )

        # Remove records without predictions
        analytics_df = analytics_df.dropna(
            subset=["Sentiment"]
        ).copy()

        if analytics_df.empty:

            st.warning(
                "There are no valid predictions available "
                "for the selected data source."
            )

        else:

            # --------------------------------------
            # 3. CALCULATE SENTIMENT COUNTS
            # --------------------------------------

            total_reviews = len(analytics_df)

            positive_count = (
                analytics_df["Sentiment"] == "Positive"
            ).sum()

            negative_count = (
                analytics_df["Sentiment"] == "Negative"
            ).sum()

            positive_percentage = (
                positive_count / total_reviews * 100
            )

            negative_percentage = (
                negative_count / total_reviews * 100
            )

            # --------------------------------------
            # 4. KPI CARDS
            # --------------------------------------

            st.subheader("Sentiment Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Reviews",
                    total_reviews
                )

            with col2:
                st.metric(
                    "Positive Reviews",
                    positive_count
                )

            with col3:
                st.metric(
                    "Negative Reviews",
                    negative_count
                )

            with col4:
                st.metric(
                    "Positive Sentiment",
                    f"{positive_percentage:.2f}%"
                )

            st.divider()

            # --------------------------------------
            # 5. DONUT CHART
            # --------------------------------------

            st.subheader("Sentiment Distribution")

            sentiment_counts = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Reviews": [
                    positive_count,
                    negative_count
                ]
            })

            donut_fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Reviews",
                hole=0.60,
                title="Positive vs Negative Reviews"
            )

            donut_fig.update_traces(
                textinfo="percent+value",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Reviews: %{value}<br>"
                    "Percentage: %{percent}"
                    "<extra></extra>"
                )
            )

            donut_fig.update_layout(
                legend_title_text="Sentiment",
                margin=dict(t=60, b=20, l=20, r=20)
            )

            st.plotly_chart(
                donut_fig,
                use_container_width=True
            )

            # --------------------------------------
            # 6. SENTIMENT COMPARISON BAR CHART
            # --------------------------------------

            st.subheader("Positive vs Negative Comparison")

            bar_fig = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Reviews",
                color="Sentiment",
                text="Reviews",
                title="Number of Reviews by Sentiment"
            )

            bar_fig.update_traces(
                textposition="outside"
            )

            bar_fig.update_layout(
                showlegend=False,
                xaxis_title="Sentiment",
                yaxis_title="Number of Reviews",
                margin=dict(t=60, b=20, l=20, r=20)
            )

            st.plotly_chart(
                bar_fig,
                use_container_width=True
            )

            # --------------------------------------
            # 7. PROBABILITY ANALYSIS
            # --------------------------------------

            st.divider()

            st.subheader("Estimated Probability Analysis")

            required_probability_columns = [
                "Negative Probability (%)",
                "Positive Probability (%)"
            ]

            if all(
                column in analytics_df.columns
                for column in required_probability_columns
            ):

                probability_df = analytics_df[
                    required_probability_columns
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                average_negative_probability = (
                    probability_df[
                        "Negative Probability (%)"
                    ].mean()
                )

                average_positive_probability = (
                    probability_df[
                        "Positive Probability (%)"
                    ].mean()
                )

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:

                    st.metric(
                        "Average Negative Probability",
                        f"{average_negative_probability:.2f}%"
                    )

                with prob_col2:

                    st.metric(
                        "Average Positive Probability",
                        f"{average_positive_probability:.2f}%"
                    )

                probability_fig = px.histogram(
                    analytics_df,
                    x="Positive Probability (%)",
                    nbins=10,
                    title="Distribution of Positive Class Probabilities",
                    labels={
                        "Positive Probability (%)":
                        "Positive Probability (%)"
                    }
                )

                st.plotly_chart(
                    probability_fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Probability information is unavailable "
                    "for this data source."
                )

            # --------------------------------------
            # 8. ANALYZED REVIEWS TABLE
            # --------------------------------------

            st.divider()

            st.subheader("Analyzed Review Records")

            st.dataframe(
                analytics_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Analytics are based on predictions available "
                "in your current application session."
            )


elif active_page == "Model Performance":

    st.title("🎯 Model Performance")
    st.write(
        "Evaluate the performance of the Bernoulli Naive Bayes "
        "sentiment classification model using its test-set results."
    )

    st.divider()

    # Model evaluation metrics
    st.subheader("Model Evaluation Metrics")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Accuracy", "79.50%")

    with metric_col2:
        st.metric("Precision", "78.64%")

    with metric_col3:
        st.metric("Recall", "81.00%")

    with metric_col4:
        st.metric("F1-Score", "79.80%")

    st.caption(
        "Evaluation results are based on the model's held-out test set "
        "containing 200 reviews."
    )

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    confusion_matrix = [
        [78, 22],
        [19, 81]
    ]

    confusion_df = pd.DataFrame(
        confusion_matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    confusion_fig = px.imshow(
        confusion_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Actual vs Predicted Sentiment"
    )

    confusion_fig.update_layout(
        xaxis_title="Predicted Sentiment",
        yaxis_title="Actual Sentiment",
        coloraxis_showscale=False,
        margin=dict(t=60, b=20, l=20, r=20)
    )

    st.plotly_chart(
        confusion_fig,
        use_container_width=True
    )

    st.divider()

    # Detailed interpretation of evaluation metrics
    st.subheader("Understanding Model Performance")

    interpretation_col1, interpretation_col2 = st.columns(2)

    with interpretation_col1:
        st.markdown("### Strengths")

        st.success(
            "**Positive Recall – 81.00%**\n\n"
            "The model correctly identifies 81 out of every "
            "100 actual positive reviews in the test set."
        )

        st.success(
            "**Overall Accuracy – 79.50%**\n\n"
            "The model correctly classifies 159 out of "
            "200 test reviews."
        )

    with interpretation_col2:
        st.markdown("### Areas for Improvement")

        st.warning(
            "**False Positives – 22 reviews**\n\n"
            "The model incorrectly classifies 22 negative "
            "reviews as positive."
        )

        st.warning(
            "**False Negatives – 19 reviews**\n\n"
            "The model incorrectly classifies 19 positive "
            "reviews as negative."
        )

    st.divider()

    st.subheader("Prediction Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Correct Predictions", "159")

    with summary_col2:
        st.metric("Incorrect Predictions", "41")

    with summary_col3:
        st.metric("Test Dataset Size", "200")


elif active_page == "About the Model":

    st.title("ℹ️ About ReviewSense AI")

    st.write(
        "ReviewSense AI is a machine learning-powered restaurant "
        "review intelligence platform that analyzes customer "
        "reviews and predicts whether their sentiment is positive "
        "or negative."
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        "The objective of this project is to demonstrate an "
        "end-to-end Natural Language Processing and Machine "
        "Learning workflow for sentiment classification."
    )

    st.write(
        "The application uses a trained Bernoulli Naive Bayes "
        "model to classify restaurant reviews into two sentiment "
        "categories: Positive and Negative."
    )

    st.divider()

    st.subheader("Project Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Project Domain**")
        st.write("Natural Language Processing (NLP)")

        st.markdown("**Problem Type**")
        st.write("Binary Sentiment Classification")

        st.markdown("**Machine Learning Algorithm**")
        st.write("Bernoulli Naive Bayes")

    with info_col2:
        st.markdown("**Application Framework**")
        st.write("Streamlit")

        st.markdown("**Feature Extraction**")
        st.write("Binary CountVectorizer")

        st.markdown("**Text Preprocessing**")
        st.write("Basic WordNet Noun Lemmatization")

    st.divider()

    st.subheader("Dataset Overview")

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

    with dataset_col1:
        st.metric("Original Records", "1,000")

    with dataset_col2:
        st.metric("Duplicate Records Removed", "4")

    with dataset_col3:
        st.metric("Final Dataset", "996")

    st.write(
        "The dataset contains restaurant reviews with two "
        "sentiment labels: 0 (Negative) and 1 (Positive)."
    )

    st.divider()

    st.subheader("Text Preprocessing Workflow")

    st.write(
        "Before classification, raw restaurant reviews pass through "
        "a text preprocessing pipeline to prepare them for feature extraction."
    )

    preprocessing_steps = [
        ("01", "Text Cleaning", "Normalize and clean raw review text."),
        ("02", "Contraction Expansion", "Expand contractions into their full forms."),
        ("03", "Stopword Handling",
         "Remove selected stopwords while preserving important sentiment words."),
        ("04", "Lemmatization", "Apply basic WordNet noun lemmatization."),
        ("05", "Feature Extraction",
         "Convert processed text into binary numerical features using CountVectorizer.")
    ]

    for number, step, description in preprocessing_steps:
        with st.container(border=True):
            step_col1, step_col2 = st.columns([1, 9])

            with step_col1:
                st.markdown(f"### {number}")

            with step_col2:
                st.markdown(f"**{step}**")
                st.write(description)

    st.divider()

    st.subheader("Model Configuration")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.markdown("**Algorithm**")
        st.write("Bernoulli Naive Bayes")

        st.markdown("**Feature Extraction**")
        st.write("Binary CountVectorizer")

        st.markdown("**Binary Feature Representation**")
        st.write("0 = Word absent, 1 = Word present")

    with model_col2:
        st.markdown("**Smoothing Parameter (Alpha)**")
        st.write("0.5")

        st.markdown("**N-gram Range**")
        st.write("(1, 1) – Unigrams")

        st.markdown("**Cross-Validation**")
        st.write("5-fold GridSearchCV")

    st.markdown("**Model Selection**")
    st.write(
        "The selected configuration uses basic WordNet noun "
        "lemmatization, binary CountVectorizer features, and "
        "Bernoulli Naive Bayes with alpha = 0.5."
    )

    st.divider()

    st.subheader("Model Limitations")

    limitations = [
        (
            "Limited Dataset Size",
            "The model was trained using 996 restaurant reviews, "
            "which may not represent the diversity of real-world reviews."
        ),
        (
            "Context Understanding",
            "The Bag-of-Words representation does not fully capture "
            "context, sarcasm, or complex sentence structures."
        ),
        (
            "Domain Dependency",
            "The model was trained on restaurant reviews and may "
            "not generalize well to reviews from unrelated domains."
        ),
        (
            "Prediction Errors",
            "The model can misclassify reviews containing mixed "
            "sentiments, subtle opinions, or unfamiliar expressions."
        ),
        (
            "Probability Calibration",
            "Predicted probabilities are model estimates and have "
            "not been independently calibrated."
        )
    ]

    for title, description in limitations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.divider()

    st.subheader("Future Improvements")

    future_improvements = [
        "Expand the training dataset with more diverse restaurant reviews.",
        "Experiment with TF-IDF and word or sentence embeddings.",
        "Explore advanced models such as Logistic Regression, "
        "Support Vector Machines, and transformer-based models.",
        "Improve the handling of sarcasm, negation, and mixed sentiments.",
        "Evaluate probability calibration and model generalization "
        "using an independent test dataset.",
        "Introduce multilingual sentiment analysis."
    ]

    for improvement in future_improvements:
        st.markdown(f"- {improvement}")
