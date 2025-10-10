from typing import Optional, List, Dict, Sequence
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors
import pandas as pd
import numpy as np
import ast
import re

## General Utilities
def generate_color_mapping_with_plotly(labels, color_scale_name="Viridis", descending=True):
    """
    Generates a dictionary mapping labels to colors using Plotly's built-in color scales.

    Parameters:
    - labels (list): Ordered list of labels (ascending or descending severity).
    - color_scale_name (str): Name of the Plotly color scale to use (e.g., "Viridis", "Plasma", "Cividis").
    - descending (bool): Whether to reverse the color scale for descending severity.

    Returns:
    - dict: Dictionary mapping labels to colors.
    """
    # Get the color scale from Plotly
    color_scale = plotly.colors.sequential.__dict__.get(color_scale_name, plotly.colors.sequential.Viridis)

    if descending:
        color_scale = color_scale[::-1]  # Reverse the color scale if descending order is required

    # Normalize the color scale to match the number of labels
    num_labels = len(labels)
    colors = [color_scale[int(i * (len(color_scale) - 1) / (num_labels - 1))] for i in range(num_labels)]

    # Map labels to colors
    color_mapping = {label: color for label, color in zip(labels, colors)}

    return color_mapping

def plot_hidden_risk_over_time(
    df: pd.DataFrame,
    time_col: str = "ADMITTIME",
    prob_prefix: str = "PROB_HIDDEN_RISK",
    event_name: str = "Event",
    title: str = "Hidden State Probabilities",
    subtitle: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,  # keys are prob column names
    save_html_file_path: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """
    Plot evolution of hidden risk probabilities over events for a single sequence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain a time column and one or more probability columns starting with `prob_prefix`.
    time_col : str
        Name of the time column (e.g., 'ADMITTIME').
    prob_prefix : str
        Prefix used to auto-detect probability columns.
    event_name : str
        X-axis label (e.g., 'Hospitalization', 'Visit', etc.).
    title : str
        Plot title.
    colors : dict[str, str] | None
        Optional mapping from probability column name -> CSS color (e.g., '#1f77b4' or 'firebrick').
        Columns without a provided color will use Plotly's default cycle.
    save_html_file_path : str | None
        If provided, saves the figure as HTML ('.html' appended if missing).
    show : bool
        If True, calls fig.show().
        
    Returns
    -------
    go.Figure
    """

    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' column not found in DataFrame.")

    # Auto-detect prob columns with the prefix
    prob_cols: Sequence[str] = [c for c in df.columns if c.startswith(prob_prefix)]
    if not prob_cols:
        raise ValueError(f"No probability columns found with prefix '{prob_prefix}'.")

    # Sort by time; tolerant to mixed/str datetimes
    _df = df.copy()
    _df[time_col] = pd.to_datetime(_df[time_col], errors="coerce")
    _df = _df.sort_values(time_col, kind="mergesort")  # stable sort to preserve ties

    # X axis: 1..N (event number)
    x_values = list(range(1, len(_df) + 1))

    # Build figure
    fig = go.Figure()

    # Ensure deterministic legend order by sorting columns
    for col in sorted(prob_cols):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=_df[col].astype(float),
                mode="lines+markers",
                name=col,  # or customize label here if desired
                line=dict(color=(colors.get(col) if colors and col in colors else None)),
            )
        )
    
    if subtitle:
        title = f"{title}<br>{subtitle}"

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title=event_name,
        yaxis_title="Probability",
        legend_title="Latent Risk State",
        template="plotly_white",
        xaxis=dict(
            tickmode="linear",
            dtick=1,       # integer ticks
            tick0=1,       # start at 1
        ),
        yaxis=dict(range=[0, 1])  # assuming probabilities in [0,1]
    )

    # Optional save
    if save_html_file_path:
        if not save_html_file_path.endswith(".html"):
            save_html_file_path = save_html_file_path + ".html"
        fig.write_html(save_html_file_path)
        print(f"Plot saved as {save_html_file_path}")

    if show:
        fig.show()

    return fig

## MIMIC Dataset

def format_feature_value(val):
    if isinstance(val, list):
        return "<br>".join(map(str, val))
    elif isinstance(val, str):
        if val.startswith("[") and val.endswith("]"):
            # Handle special cases like numpy string lists
            try:
                evaluated = ast.literal_eval(val)
                if isinstance(evaluated, (list, tuple, np.ndarray)):
                    return "<br>".join(map(str, evaluated))
            except Exception:
                pass
    elif pd.isnull(val):
        return "NA"
    elif isinstance(val, float):
        return f"{val:.2f}"
    return str(val)

def plot_subject_evolution(df, subject_id,
                           features_to_plot: Optional[list] = None,
                           save_html_file_path: Optional[str] = None,
                           textposition='outside', extend_time_horizon_by=365, show: bool = False):
    # Filter for the patient
    patient_df = df[df['SUBJECT_ID'] == subject_id].copy()
    patient_df = patient_df.sort_values('ADMITTIME')

    # Features to track over time (excluding ID/time columns)
    if features_to_plot is None:
        # Default features to plot if not provided
        features_to_plot = [
            'HOSPITALIZATION_DAYS', 'NUM_COMORBIDITIES',
            'NUM_PREV_HOSPITALIZATIONS', 'DAYS_SINCE_LAST_HOSPITALIZATION',
            'DAYS_IN_ICU', 'NUM_DRUGS', 'NUM_PROCEDURES'
        ]

    # Melt the data so each feature is a row
    melted = patient_df.melt(
        id_vars=['ADMITTIME', 'ADMISSION_TYPE', 'DISCHTIME'],
        value_vars=features_to_plot,
        var_name='Feature',
        value_name='Value'
    )

    # Convert all values to string for display
    melted['Value'] = melted['Value'].apply(format_feature_value)

    # Define colors for admission types
    color_discrete_map = {
        "URGENT": "red",
        "EMERGENCY": "orange",
        "ELECTIVE": "green"
    }

    # Create the plot
    fig = px.timeline(
        melted,
        x_start='ADMITTIME',
        x_end='DISCHTIME',
        y='Feature',
        color='ADMISSION_TYPE',
        text='Value',
        title=f'Evolution of SUBJECT_ID {subject_id}',
        color_discrete_map=color_discrete_map
    )
    fig.update_traces(textposition=textposition, textfont_size=10)
    # Adjust layout to avoid text being cut off
    fig.update_layout(
        height=600,
        yaxis_title='Feature',
        xaxis_title='Time',
        margin=dict(l=100, r=100, t=50, b=50),
        xaxis=dict(range=[melted['ADMITTIME'].min() - pd.Timedelta(days=100), melted['DISCHTIME'].max() + pd.Timedelta(days=extend_time_horizon_by)])
    )

    if save_html_file_path:
        if not save_html_file_path.endswith('.html'):
            save_html_file_path += '.html'
        # Save the plot as a html image
        fig.write_html(save_html_file_path)
        print(f"Plot saved as {save_html_file_path}")

    if show:
        fig.show()

    return fig

def plot_hidden_risk_states_patient(patient_sequence, labels, colors, save_html_file_path: Optional[str] = None):
    # Create the line plot
    fig = go.Figure()
    x_values = list(range(len(patient_sequence)))  # X axis: 0, 1, 2, ...

    for state_idx, state_label in labels.items():
        probs = [point[state_idx] for point in patient_sequence]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=probs,
            mode='lines+markers',
            name=state_label,
            line=dict(color=colors[state_label]),
        ))

    # Update layout
    fig.update_layout(
        title="Hidden State Probabilities",
        xaxis_title="Hospitalization",
        yaxis_title="Probability",
        legend_title="Latent Risk State",
        template="plotly_white",
        xaxis=dict(
            tickmode='linear',  # Ensures ticks are placed linearly
            dtick=1,            # Step size for ticks is 1 (i.e., integers only)
            tick0=0             # Start ticks at 0
        )
    )

    if save_html_file_path:
        if not save_html_file_path.endswith('.html'):
            save_html_file_path = save_html_file_path + '.html'
        # Save the plot as a html image
        fig.write_html(save_html_file_path)
        print(f"Plot saved as {save_html_file_path}")

    # Show the plot
    fig.show()

def plot_patients_hospitalizations(df, subject_ids: Optional[list] = None,
                                    additional_hover_cols: Optional[list] = None,
                                    color_map: Optional[dict] = None,
                                    color_col: Optional[str] = None,
                                    max_number_days: Optional[int] = None,
                                    save_html_file_path: Optional[str] = None):
    """
    Plot normalized hospitalization timelines for multiple patients, with color legend.

    Parameters:
    - df: DataFrame with 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME' (datetime columns).
    - subject_ids: Optional list of SUBJECT_IDs to include.
    - additional_hover_cols: Additional columns for hover text.
    - color_map: Dict mapping color_col values to specific colors.
    - color_col: Column name to group and color traces by.
    """
    fig = go.Figure()

    if subject_ids is None:
        subject_ids = df['SUBJECT_ID'].unique()

    y_labels = {subject_id: i for i, subject_id in enumerate(subject_ids)}

    # Track which color group has already been added to the legend
    shown_legends = set()

    for subject_id in subject_ids:
        patient_df = df[df['SUBJECT_ID'] == subject_id].copy()
        patient_df = patient_df.sort_values('ADMITTIME')
        origin = patient_df['ADMITTIME'].min()

        for _, row in patient_df.iterrows():
            start_day = (row['ADMITTIME'] - origin).days
            end_day = (row['DISCHTIME'] - origin).days
            death_day = (row['DOD'] - origin).days if 'DOD' in row and pd.notna(row['DOD']) else None

            hover_text = (
                f"Patient ID: {subject_id}<br>"
                f"Relative Day: {start_day} to {end_day}<br>"
                f"Admit: {row['ADMITTIME']}<br>"
                f"Discharge: {row['DISCHTIME']}"
            )

            if additional_hover_cols:
                for col in additional_hover_cols:
                    hover_text += f"<br>{col}: {row[col]}"

            # Determine color
            if color_col and color_map and row[color_col] in color_map:
                value = row[color_col]
                line_color = color_map[value]
                show_legend = value not in shown_legends
                legend_name = str(value)
                shown_legends.add(value)
            else:
                line_color = 'black'
                show_legend = False
                legend_name = None

            fig.add_trace(go.Scatter(
                x=[start_day, end_day],
                y=[y_labels[subject_id]] * 2,
                mode='lines',
                line=dict(width=10, color=line_color),
                name=legend_name,
                hoverinfo='text',
                showlegend=show_legend,
                text=hover_text
            ))

            if death_day is not None:
                show_legend = 'Death' not in shown_legends
                fig.add_trace(go.Scatter(
                    x=[death_day],
                    y=[y_labels[subject_id]],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='black'),
                    name='Death',
                    hoverinfo='text',
                    text=f"Death: {row['DOD']}",
                    showlegend=show_legend
                ))
                shown_legends.add('Death')

    xaxis_dict = dict(
            tickmode='linear',
            tickformat='%Y-%m-%d',
            tickfont=dict(size=10),  # Adjust font size for x-axis labels
            dtick=60
    )

    if max_number_days is not None:
        xaxis_dict['range'] = [0, max_number_days]
        xaxis_dict['dtick'] = max_number_days // 10  # Adjust tick interval based on max_number_days

    fig.update_layout(
        title='Normalized Hospitalization Timelines (Relative to First Admission)',
        xaxis_title='Days Since First Admission',
        yaxis_title='Patient',
        yaxis=dict(
            tickmode='array',
            tickvals=list(y_labels.values()),
            ticktext=[f'SUBJECT_ID {id}' for id in subject_ids],
        ),
        height=400 + 30 * len(subject_ids),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest',
        showlegend=True
    )

    fig.update_xaxes(**xaxis_dict)

    if save_html_file_path:
        if not save_html_file_path.endswith('.html'):
            save_html_file_path += '.html'
        # Save the plot as a html image
        fig.write_html(save_html_file_path)
        print(f"Plot saved as {save_html_file_path}")

    fig.show()

## Drug Relapse Dataset

def plot_drug_history_of_a_donor(donor_id: str, drug_tests_df: pd.DataFrame, result_col: str = 'drug_test_positive',
                                            time_col: str = 'time', donor_id_col: str = 'donor_id', drug_class_col: str = 'drug_class'):
    donor_df = drug_tests_df[drug_tests_df[donor_id_col] == donor_id].copy()

    # Ensure that ScheduledDate is in datetime format
    donor_df[time_col] = pd.to_datetime(donor_df[time_col])

    # Sort the DataFrame by ScheduledDate
    donor_df = donor_df.sort_values(by=time_col)

    donor_df = donor_df.groupby([donor_id_col, drug_class_col, time_col]).agg({"drug_test_positive": "any"}).reset_index()

    # Create separate DataFrames for positive, negative, and NaN events
    positive_events = donor_df[donor_df[result_col] == 1]
    negative_events = donor_df[donor_df[result_col] == 0]
    nan_events = donor_df[donor_df[result_col].isna()]

    # Create scatter plots for positive, negative, and NaN events
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=negative_events[time_col],
        y=negative_events[drug_class_col],
        mode='markers',
        marker=dict(color='darkgreen', size=12),
        name='Negative'
    ))

    fig.add_trace(go.Scatter(
        x=positive_events[time_col],
        y=positive_events[drug_class_col],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Positive'
    ))

    fig.add_trace(go.Scatter(
        x=nan_events[time_col],
        y=nan_events[drug_class_col],
        mode='markers',
        marker=dict(color='gray', size=12),
        name='NaN'
    ))

    # Update the layout to customize the font of the x and y axis titles
    fig.update_layout(
    title=f'Events for Donor {donor_id}',
    xaxis_title='Time',
    yaxis_title='Drug Class',
    showlegend=True,
    xaxis=dict(
        tickfont=dict(size=14),  # Increase the size of the x-axis labels
        title=dict(
            text='Time',
            font=dict(size=16, family='Arial', color='blue')  # Customize the font of the x-axis title
        )
    ),
    yaxis=dict(
        tickfont=dict(size=14),  # Increase the size of the y-axis labels
        title=dict(
            text='Drug Class',
            font=dict(size=16, family='Arial', color='blue')  # Customize the font of the y-axis title
        )
        )
    )

    # Show the plot
    fig.show()

def plot_positive_dates_timeline_of_a_donor(donor_id: str, drug_tests_df: pd.DataFrame,
                                            result_col: str = 'drug_test_positive',
                                            time_col: str = 'time', donor_id_col: str = 'donor_id'):
    donor_df = drug_tests_df[drug_tests_df[donor_id_col] == donor_id].copy()
    # Ensure 'time' column is in datetime format
    donor_df[time_col] = pd.to_datetime(donor_df[time_col])

    donor_df = donor_df.groupby(time_col).agg(positive_date=(result_col, "any")).reset_index()

    # Separate data based on result values
    positive_events = donor_df[donor_df['positive_date'] == 1]
    negative_events = donor_df[donor_df['positive_date'] == 0]

    # Create a plotly figure
    fig = go.Figure()

    # Add red dots for result = 0
    fig.add_trace(go.Scatter(
        x=negative_events[time_col],
        y=[1] * len(negative_events),  # Use a constant y-value for timeline
        mode='markers',
        marker=dict(color='green', size=10),
        name='Negative'
    ))

    # Add green dots for result = 1
    fig.add_trace(go.Scatter(
        x=positive_events[time_col],
        y=[1] * len(positive_events),  # Use a constant y-value for timeline
        mode='markers',
        marker=dict(color='red', size=10),
        name='Positive'
    ))

    # Update layout
    fig.update_layout(
        title=f'Timeline of Results of Donor {donor_id}',
        xaxis_title='Time',
        yaxis_title='',
        yaxis=dict(showticklabels=False),  # Hide y-axis labels
        showlegend=True
    )

    # Show the plot
    fig.show()

def parse_numpy_str_list(val):
    # e.g. "['Alcohol' 'Amphetamines' 'Benzodiazepines']"
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        # Remove the brackets
        inside = val[1:-1].strip()
        # Find words inside single or double quotes
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", inside)
        # items is a list of tuples; pick non-empty part
        drugs = [i[0] or i[1] for i in items if (i[0] or i[1])]
        if not drugs:  # fallback: split by space
            drugs = [x.strip("'\"") for x in inside.split() if x.strip()]
        return drugs
    else:
        return val  # not a numpy string-list, return as is

def plot_donor_relapse_evolution(
    df: pd.DataFrame,
    donor_id,
    features_to_plot: Optional[List[str]] = None,
    save_html_file_path: Optional[str] = None,
    textposition: str = 'outside',
    extend_time_horizon_by: int = 30,
):
    # Filter for the donor
    donor_df = df[df['DONOR_ID'] == donor_id].copy()
    donor_df = donor_df.sort_values('RELAPSE_START')

    # Features to track over time
    if features_to_plot is None:
        features_to_plot = [
            'PREV_POSITIVE_DRUGS',
            'DRUGS_TESTED',
            'NUM_POSITIVES_SINCE_LAST_NEGATIVE',
            'NUM_TESTS_PERIOD',
            'TIME_SINCE_LAST_POSITIVE',
            'TIME_UNTIL_NEXT_POSITIVE',
            'TIME_SINCE_LAST_NEGATIVE',
        ]
    
    if 'DRUGS_TESTED' in features_to_plot:
        donor_df["DRUGS_TESTED"] = donor_df["DRUGS_TESTED"].map(parse_numpy_str_list)


    # Melt the data so each feature is a row
    melted = donor_df.melt(
        id_vars=['RELAPSE_START', 'RELAPSE_END'],
        value_vars=features_to_plot,
        var_name='Feature',
        value_name='Value'
    )

    # Convert all values to string for display
    melted['Value'] = melted['Value'].apply(format_feature_value)

    fig = px.timeline(
        melted,
        x_start='RELAPSE_START',
        x_end='RELAPSE_END',
        y='Feature',
        text='Value',
        title=f'Relapse Evolution of DONOR_ID {donor_id}'
    )

    fig.update_traces(textposition=textposition, textfont_size=10)

    # Adjust layout to avoid text being cut off
    fig.update_layout(
        height=600,
        yaxis_title='Feature',
        xaxis_title='Time',
        margin=dict(l=100, r=100, t=50, b=50),
        xaxis=dict(
            range=[
                melted['RELAPSE_START'].min() - pd.Timedelta(days=100),
                melted['RELAPSE_END'].max() + pd.Timedelta(days=extend_time_horizon_by)
            ]
        ),
    )

    if save_html_file_path:
        if not save_html_file_path.endswith('.html'):
            save_html_file_path += '.html'
        fig.write_html(save_html_file_path)
        print(f"Plot saved as {save_html_file_path}")

    fig.show()