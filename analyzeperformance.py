import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# File Paths
current_quiz_file = "/content/LLQT.json"
quiz_submission_file = "/content/rJvd7g.json"
historical_quiz_file = "/content/XgAgFJ.json"

# 1. Data Loading and Cleaning
# Load JSON files
with open(current_quiz_file, 'r') as file:
    current_quiz_data = json.load(file)

with open(quiz_submission_file, 'r') as file:
    quiz_submission_data = json.load(file)

with open(historical_quiz_file, 'r') as file:
    historical_quiz_data = json.load(file)

# Normalize and Clean Data
current_quiz_df = pd.json_normalize(current_quiz_data["quiz"]["questions"])
submission_df = pd.DataFrame([quiz_submission_data])
historical_df = pd.json_normalize(historical_quiz_data)

historical_df["accuracy"] = historical_df["accuracy"].str.replace(" %", "").astype(float)
historical_df["score"] = pd.to_numeric(historical_df["score"], errors='coerce')
historical_df["correct_answers"] = pd.to_numeric(historical_df["correct_answers"], errors='coerce')
historical_df["incorrect_answers"] = pd.to_numeric(historical_df["incorrect_answers"], errors='coerce')

# Check for missing values
print("\nMissing Values in Historical Quiz Data:")
print(historical_df.isnull().sum())

# 2. Analyze Student Performance
def analyze_current_quiz(submission_data):
    total_questions = submission_data["total_questions"]
    correct_answers = submission_data["correct_answers"]
    incorrect_answers = submission_data["incorrect_answers"]
    accuracy = float(submission_data["accuracy"].replace(" %", ""))

    print("=== Current Quiz Performance ===")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Incorrect Answers: {incorrect_answers}")
    print(f"Accuracy: {accuracy}%")
    print()
    return accuracy

def analyze_topic_performance(historical_df):
    topic_analysis = (historical_df.groupby("quiz.topic").agg(
            avg_accuracy=("accuracy", "mean"),
            avg_score=("score", "mean"),
            total_correct=("correct_answers", "sum"),
            total_incorrect=("incorrect_answers", "sum"))
        .reset_index())
    print("=== Topic-Wise Performance ===")
    print(topic_analysis)
    print()
    return topic_analysis

current_accuracy = analyze_current_quiz(quiz_submission_data)
topic_performance = analyze_topic_performance(historical_df)

# 3. ML-Based Performance Trend Prediction
def predict_performance_trend(historical_df):
    historical_df["submission_date"] = pd.to_datetime(historical_df["submitted_at"])
    historical_df["date_numeric"] = historical_df["submission_date"].map(pd.Timestamp.toordinal)

    X = historical_df[["date_numeric"]]
    y = historical_df["accuracy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nPredicted Accuracy for Future Quizzes:")
    print(predictions)

predict_performance_trend(historical_df)

# 4. Generate Insights and Recommendations
def generate_insights(topic_performance, current_accuracy):
    weak_topics = topic_performance[topic_performance["avg_accuracy"] < 50]
    strong_topics = topic_performance[topic_performance["avg_accuracy"] >= 75]

    insights = {
        "current_accuracy": current_accuracy,
        "weak_topics": weak_topics[["quiz.topic", "avg_accuracy", "total_incorrect"]].to_dict(orient="records"),
        "strong_topics": strong_topics[["quiz.topic", "avg_accuracy", "total_correct"]].to_dict(orient="records"),
    }
    return insights

def generate_recommendations(insights):
    recommendations = []

    for topic in insights["weak_topics"]:
        recommendations.append(
            f"Focus on revising '{topic['quiz.topic']}' (accuracy = {topic['avg_accuracy']}%). "
            f"Practice more questions to reduce incorrect answers ({topic['total_incorrect']})."
        )
    for topic in insights["strong_topics"]:
        recommendations.append(
            f"Keep excelling in '{topic['quiz.topic']}' (accuracy = {topic['avg_accuracy']}%). "
            "Try advanced-level questions to further improve."
        )
    return recommendations

insights = generate_insights(topic_performance, current_accuracy)
recommendations = generate_recommendations(insights)

# 5. Define Student Persona
def define_student_persona(insights, historical_df):
    avg_accuracy = historical_df["accuracy"].mean()
    avg_score = historical_df["score"].mean()
    recent_accuracy = historical_df.sort_values("submitted_at", ascending=False).iloc[0]["accuracy"]

    if avg_accuracy >= 80 and recent_accuracy >= 80:
        persona = "Consistent Performer"
        description = (
            "You consistently achieve high accuracy and scores across all topics. "
            "Keep up the excellent work, and consider challenging yourself with more advanced questions."
        )
    elif avg_accuracy >= 50 and len(insights["weak_topics"]) > 0:
        persona = "Focused Improver"
        description = (
            "You have a solid foundation, but some weak areas need improvement. "
            "Focus on strengthening your weak topics to achieve greater overall consistency."
        )
    elif avg_accuracy < 50:
        persona = "Beginner Learner"
        description = (
            "Your performance indicates you are in the early stages of learning. "
            "Focus on understanding fundamental concepts and practicing regularly to improve accuracy and scores."
        )
    elif recent_accuracy > avg_accuracy:
        persona = "Steady Improver"
        description = (
            "Your recent quizzes show improvement over time. "
            "Keep practicing and working on weak areas to maintain this positive trend."
        )
    else:
        persona = "Balanced Learner"
        description = (
            "You have a balanced performance with strengths in some topics. "
            "Focus on maintaining your strong areas while addressing weaker ones."
        )

    print("=== Student Persona ===")
    print(f"Persona: {persona}")
    print(f"Description: {description}")

    return {"persona": persona, "description": description}

student_persona = define_student_persona(insights, historical_df)

# 6. Visualizations
def visualize_topic_performance(topic_performance):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=topic_performance, x="quiz.topic", y="avg_accuracy", palette="Blues_d")
    plt.xlabel("Topics")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Topic-Wise Performance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

visualize_topic_performance(topic_performance)
