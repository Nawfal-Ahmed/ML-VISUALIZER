from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # New Import
from sklearn.preprocessing import PolynomialFeatures # New Import
from sklearn.pipeline import make_pipeline # New Import
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
import warnings

# Suppress LogisticRegression ConvergenceWarning for small, synthetic dataset
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Sample Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]) # Extended X for better visualization
y_linear = np.array([2, 3, 5, 7, 11, 13, 17, 19]) # Simple linear data
# Data for non-linear fit (e.g., quadratic-like)
y_poly = np.array([1, 2, 4, 7, 11, 16, 22, 29]) 
# Binary classification data
y_logistic = np.array([0, 0, 0, 1, 1, 1, 1, 1]) 

# Sklearn models require X to be 2D: [n_samples, n_features]
X_reshaped = X.reshape(-1, 1)
# Create a dense array of x values for smoother curve plotting (e.g., for polynomial)
X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

@app.route("/")
def home():
    """Renders the main dashboard page."""
    return render_template("index.html")

@app.route("/get_data", methods=["POST"])
def get_data():
    """Calculates and returns regression data and metrics using scikit-learn."""
    regression_type = request.json.get("type")

    # --- Linear Regression (unchanged) ---
    if regression_type == "linear":
        model = LinearRegression()
        model.fit(X_reshaped, y_linear)
        y_pred = model.predict(X_reshaped)

        mse = mean_squared_error(y_linear, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_linear, y_pred)

        return jsonify({
            "x": X.tolist(),
            "y": y_linear.tolist(),
            "fit": model.predict(X_smooth).tolist(), # Use smooth curve for fit
            "x_fit": X_smooth.flatten().tolist(), # New x values for the fit
            "description": "Linear Regression (sklearn) fits a straight line: $Y = mx + c$.",
            "metrics": {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R²": round(r2, 4),
                "Slope ($m$)": round(model.coef_[0], 4),
                "Intercept ($c$)": round(model.intercept_, 4)
            },
            "model_type": "Linear"
        })

    # --- Polynomial Regression (NEW) ---
    elif regression_type == "polynomial":
        # Create a pipeline for 2nd-degree polynomial fit
        # We need the pipeline because prediction on X_smooth needs the same transformation
        degree = 2
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_reshaped, y_poly)
        
        y_pred = model.predict(X_reshaped)
        y_smooth_pred = model.predict(X_smooth)

        mse = mean_squared_error(y_poly, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_poly, y_pred)
        
        # The coefficients are tricky to extract and interpret from the pipeline,
        # so we'll show the standard regression metrics.

        return jsonify({
            "x": X.tolist(),
            "y": y_poly.tolist(),
            "fit": y_smooth_pred.tolist(), # Use smooth curve for fit
            "x_fit": X_smooth.flatten().tolist(),
            "description": f"Polynomial Regression (Degree {degree}) fits a curve to non-linear data.",
            "metrics": {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R²": round(r2, 4),
            },
            "model_type": "Polynomial"
        })

    # --- Logistic Regression (updated with smooth fit) ---
    elif regression_type == "logistic":
        model = LogisticRegression(solver='liblinear', random_state=42) 
        model.fit(X_reshaped, y_logistic)
        
        # Predict probabilities on smooth data for a better sigmoid curve
        y_pred_prob_smooth = model.predict_proba(X_smooth)[:, 1]
        
        # Predict classes on original data for accuracy
        y_pred_class = model.predict(X_reshaped)
        
        accuracy = accuracy_score(y_logistic, y_pred_class)
        logloss = log_loss(y_logistic, model.predict_proba(X_reshaped)[:, 1])

        return jsonify({
            "x": X.tolist(),
            "y": y_logistic.tolist(),
            "fit": y_pred_prob_smooth.tolist(),
            "x_fit": X_smooth.flatten().tolist(),
            "description": "Logistic Regression (sklearn) models the probability of a binary outcome (sigmoid function).",
            "metrics": {
                "Accuracy": round(accuracy, 4),
                "Log Loss": round(logloss, 4),
                "Coefficient ($b_1$)": round(model.coef_[0][0], 4),
                "Intercept ($b_0$)": round(model.intercept_[0], 4)
            },
            "model_type": "Logistic"
        })

    # --- K-Nearest Neighbors Classifier (NEW) ---
    elif regression_type == "knn":
        n_neighbors = 3
        model = KNeighborsClassifier(n_neighbors=n_neighbors) 
        model.fit(X_reshaped, y_logistic)
        
        # Predict probabilities on smooth data for a better probability map
        y_pred_prob_smooth = model.predict_proba(X_smooth)[:, 1]
        
        # Predict classes on original data for accuracy
        y_pred_class = model.predict(X_reshaped)
        
        accuracy = accuracy_score(y_logistic, y_pred_class)
        
        return jsonify({
            "x": X.tolist(),
            "y": y_logistic.tolist(),
            # KNN probability output is blocky, but we use smooth for visualization
            "fit": y_pred_prob_smooth.tolist(), 
            "x_fit": X_smooth.flatten().tolist(),
            "description": f"K-Nearest Neighbors (k={n_neighbors}) predicts class based on the majority class of its 'k' closest neighbors.",
            "metrics": {
                "Accuracy": round(accuracy, 4),
                "k (Neighbors)": n_neighbors,
            },
            "model_type": "KNN"
        })

    return jsonify({"error": "Invalid regression type"}), 400

if __name__ == "__main__":
    app.run(debug=True)