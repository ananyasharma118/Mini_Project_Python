# ==========================================================
# PROJECT: STUDY HOURS VS MARKS ANALYZER
# B.Tech 1st Year Mini Project
# Technologies: Python, NumPy, Pandas, Matplotlib, Ipywidgets
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, clear_output
from google.colab import files

# ----------------------------------------------------------
# 1. SAMPLE DATA FOR TRAINING THE PREDICTION MODEL
# ----------------------------------------------------------

np.random.seed(42)

study_data = np.array([
    [2, 35],
    [4, 42],
    [6, 48],
    [8, 55],
    [10, 60],
    [12, 65],
    [15, 70],
    [18, 75],
    [20, 80],
    [25, 85],
    [30, 90],
    [35, 95]
])

X = study_data[:, 0]
Y = study_data[:, 1]

# Linear regression using NumPy
slope, intercept = np.polyfit(X, Y, 1)

def predict_marks(total_hours):
    prediction = slope * total_hours + intercept
    return np.clip(prediction, 0, 100)


# ----------------------------------------------------------
# 2. STORE STUDENT RECORDS
# ----------------------------------------------------------

records = []

# ----------------------------------------------------------
# 3. CREATE DASHBOARD HEADING
# ----------------------------------------------------------

heading = widgets.HTML("""
<div style="
background: linear-gradient(120deg,#30105e,#7352c7,#4285d4);
padding:25px;
border-radius:15px;
text-align:center;
color:white;">

<h1 style="margin:0;">STUDY SMART 📚</h1>
<h2 style="margin:8px;">Study Hours vs Marks Analyzer</h2>
<p>Track your effort. Analyze your performance.</p>
</div>
""")

# ----------------------------------------------------------
# 4. INPUT FIELDS
# ----------------------------------------------------------

name = widgets.Text(
    description="Name:",
    placeholder="Enter your name",
    style={'description_width':'initial'},
    layout=widgets.Layout(width='95%')
)

subject = widgets.Text(
    description="Subject:",
    placeholder="Enter subject name",
    style={'description_width':'initial'},
    layout=widgets.Layout(width='95%')
)

hours = widgets.BoundedFloatText(
    description="Daily Study Hours:",
    value=4,
    min=0,
    max=24,
    step=0.5,
    style={'description_width':'initial'},
    layout=widgets.Layout(width='95%')
)

days = widgets.BoundedIntText(
    description="Days Studied:",
    value=10,
    min=1,
    max=365,
    style={'description_width':'initial'},
    layout=widgets.Layout(width='95%')
)

total_marks = widgets.Dropdown(
    options=[50, 75, 100],
    value=100,
    description="Exam Marks:",
    style={'description_width':'initial'},
    layout=widgets.Layout(width='95%')
)

# ----------------------------------------------------------
# 5. BUTTONS
# ----------------------------------------------------------

analyze_btn = widgets.Button(
    description="ANALYZE MY PERFORMANCE",
    button_style='success',
    icon='bar-chart',
    layout=widgets.Layout(width='100%', height='45px')
)

history_btn = widgets.Button(
    description="VIEW PREVIOUS RECORDS",
    button_style='info',
    icon='history'
)

clear_btn = widgets.Button(
    description="CLEAR INPUTS",
    button_style='warning',
    icon='refresh'
)

download_btn = widgets.Button(
    description="DOWNLOAD CSV REPORT",
    button_style='primary',
    icon='download'
)

# Output areas
result_output = widgets.Output()
graph_output = widgets.Output()
history_output = widgets.Output()

# ----------------------------------------------------------
# 6. PERFORMANCE CATEGORY
# ----------------------------------------------------------

def get_category(percentage):

    if percentage >= 85:
        return "EXCELLENT 🌟", "Keep up your wonderful effort!"

    elif percentage >= 70:
        return "VERY GOOD 🎯", "You are making great progress!"

    elif percentage >= 50:
        return "AVERAGE 📖", "A little more consistency can help!"

    else:
        return "NEEDS IMPROVEMENT 💡", "Keep learning and improving!"

# ----------------------------------------------------------
# 7. ANALYZE BUTTON FUNCTION
# ----------------------------------------------------------

def analyze_performance(button):

    with result_output:
        clear_output(wait=True)

        student = name.value.strip()
        sub = subject.value.strip()

        if not student or not sub:
            print("⚠️ Please enter your name and subject.")
            return

        daily_hours = hours.value
        study_days = days.value
        maximum_marks = total_marks.value

        total_hours = daily_hours * study_days

        # Predict percentage using regression
        predicted_percentage = predict_marks(total_hours)

        # Convert percentage into examination marks
        predicted_score = round(
            predicted_percentage * maximum_marks / 100, 2
        )

        category, message = get_category(predicted_percentage)

        # Save record
        record = {
            "Student Name": student,
            "Subject": sub,
            "Daily Hours": daily_hours,
            "Days Studied": study_days,
            "Total Study Hours": total_hours,
            "Exam Total Marks": maximum_marks,
            "Predicted Marks": predicted_score,
            "Predicted Percentage": round(predicted_percentage, 2),
            "Performance": category
        }

        records.append(record)

        # Beautiful result card
        display(widgets.HTML(f"""
        <div style="
        background:linear-gradient(120deg,#271052,#6545ad);
        color:white;
        padding:25px;
        border-radius:15px;
        text-align:center;
        margin-top:10px;">

        <h2>YOUR PERFORMANCE REPORT 🎓</h2>

        <h3>{student} | {sub}</h3>

        <h1 style="font-size:48px;margin:10px;">
        {predicted_score} / {maximum_marks}
        </h1>

        <h2>{predicted_percentage:.1f}%</h2>

        <h3>{category}</h3>

        <p>Total Study Hours: {total_hours}</p>

        <p>{message}</p>
        </div>

        <p style="color:gray;text-align:center;">
        Note: This is an illustrative prediction based on sample data,
        not a guarantee of actual examination marks.
        </p>
        """))

    # Draw graphs
    with graph_output:
        clear_output(wait=True)

        df = pd.DataFrame(records)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Graph 1: Study Hours vs Predicted Percentage
        sample_hours = np.linspace(0, 60, 100)

        axes[0].scatter(
            X, Y,
            color='purple',
            label='Training Data',
            s=60
        )

        axes[0].plot(
            sample_hours,
            predict_marks(sample_hours),
            color='orange',
            linewidth=2,
            label='Prediction Line'
        )

        axes[0].scatter(
            total_hours,
            predicted_percentage,
            color='red',
            s=130,
            marker='*',
            label='Your Result'
        )

        axes[0].set_title("Study Hours vs Predicted Marks")
        axes[0].set_xlabel("Total Study Hours")
        axes[0].set_ylabel("Predicted Percentage")
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Graph 2: Performance History
        axes[1].bar(
            df["Student Name"],
            df["Predicted Percentage"],
            color='mediumpurple'
        )

        axes[1].set_title("Performance History")
        axes[1].set_ylabel("Predicted Percentage (%)")
        axes[1].set_ylim(0, 105)
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Progress visualization
        plt.figure(figsize=(9, 2))

        plt.barh(
            ["Your Predicted Score"],
            [predicted_percentage],
            color='mediumpurple'
        )

        plt.xlim(0, 100)
        plt.xlabel("Percentage")
        plt.title("Your Performance Progress")
        plt.grid(axis='x', alpha=0.3)
        plt.show()


# ----------------------------------------------------------
# 8. VIEW PREVIOUS RECORDS
# ----------------------------------------------------------

def show_history(button):

    with history_output:
        clear_output(wait=True)

        if len(records) == 0:
            print("No records available yet!")
        else:
            df = pd.DataFrame(records)

            display(widgets.HTML("""
            <h2 style="color:#6545ad;">
            📚 YOUR STUDY HISTORY
            </h2>
            """))

            display(df)


# ----------------------------------------------------------
# 9. CLEAR INPUTS
# ----------------------------------------------------------

def clear_inputs(button):

    name.value = ""
    subject.value = ""
    hours.value = 4
    days.value = 10
    total_marks.value = 100

    with result_output:
        clear_output()

    with graph_output:
        clear_output()

    with history_output:
        clear_output()


# ----------------------------------------------------------
# 10. DOWNLOAD CSV FILE
# ----------------------------------------------------------

def download_report(button):

    if len(records) == 0:
        with history_output:
            clear_output()
            print("Please analyze at least one record first!")
        return

    df = pd.DataFrame(records)

    filename = "Study_Performance_Report.csv"

    df.to_csv(filename, index=False)

    files.download(filename)


# ----------------------------------------------------------
# 11. CONNECT BUTTONS TO FUNCTIONS
# ----------------------------------------------------------

analyze_btn.on_click(analyze_performance)
history_btn.on_click(show_history)
clear_btn.on_click(clear_inputs)
download_btn.on_click(download_report)


# ----------------------------------------------------------
# 12. DISPLAY COMPLETE DASHBOARD
# ----------------------------------------------------------

input_panel = widgets.VBox([
    widgets.HTML("<h3 style='color:#6545ad;'>📝 ENTER YOUR DETAILS</h3>"),
    name,
    subject,
    hours,
    days,
    total_marks,
    analyze_btn,
    widgets.HBox([clear_btn, history_btn, download_btn])
], layout=widgets.Layout(
    width='100%',
    padding='15px'
))

dashboard = widgets.VBox([
    heading,
    input_panel,
    result_output,
    graph_output,
    history_output
])

display(dashboard)

print("🎉 Study Hours vs Marks Analyzer is Ready!")