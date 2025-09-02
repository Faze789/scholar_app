from flask import Flask, request, jsonify
import pandas as pd
import re
import numpy as np
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# ========== HELPER FUNCTIONS ==========
def calculate_aggregate(matric_marks, fsc_marks, test_marks, totals, weights):
    """Calculate weighted aggregate score"""
    matric_pct = (matric_marks / totals["matric"]) * 100 if totals.get("matric") else 0
    fsc_pct = (fsc_marks / totals["fsc"]) * 100 if totals.get("fsc") else 0
    test_pct = (test_marks / totals["test"]) * 100 if totals.get("test") and test_marks is not None else 0

    aggregate = (matric_pct * weights.get("matric", 0)) + \
                (fsc_pct * weights.get("fsc", 0)) + \
                (test_pct * weights.get("test", 0))
    return aggregate

def normalize(text):
    """Normalize program names for comparison"""
    if pd.isna(text):
        return ""
    return re.sub(r'\b(bs|bsc|bachelors|in|science|scien)\b', '', str(text).lower()).strip()

def extract_year(year_str):
    """Extract year from string"""
    match = re.search(r'(\d{4})', str(year_str).replace(',', ''))
    return int(match.group(1)) if match else None

def predict_cutoff(X, y, target_year):
    """Predict cutoff using linear regression"""
    if len(X) == 0:
        return None
        
    if X.mean() > 1900:  # If using actual years
        pred_input = target_year
    else:  # If using indices
        pred_input = len(X)
    
    model = LinearRegression()
    model.fit(X, y)
    return float(model.predict(np.array([[pred_input]]))[0])

def predict_with_single_point(y_value, target_year, current_year, trend="stable"):
    """Predict cutoff when we only have one data point"""
    if trend == "increasing":
        # Assume a small annual increase (1% per year)
        years_diff = target_year - current_year
        return y_value + (years_diff * 0.5)
    elif trend == "decreasing":
        # Assume a small annual decrease (0.5% per year)
        years_diff = target_year - current_year
        return y_value - (years_diff * 0.3)
    else:
        # Assume stable cutoffs
        return y_value

def get_latest_cutoff(df, program, cutoff_col, year_col=None, program_col="Program"):
    """Get the latest cutoff for a program"""
    df = df.copy()
    if program_col not in df.columns:
        return None, None
        
    df["Program_norm"] = df[program_col].apply(normalize)
    program_norm = normalize(program)
    matched = df[df["Program_norm"].str.contains(program_norm, na=False)]
    
    if matched.empty:
        return None, None

    if year_col and year_col in df.columns:
        matched = matched.copy()
        matched.loc[:, 'Year_Num'] = matched[year_col].apply(extract_year)
        matched = matched.dropna(subset=['Year_Num'])
        if not matched.empty:
            latest = matched.loc[matched['Year_Num'].idxmax()]
            return float(latest[cutoff_col]), int(latest['Year_Num'])
    
    val = matched.iloc[0].get(cutoff_col, None)
    return (float(val), None) if pd.notna(val) else (None, None)

def prepare_training_data(df, program, year_col, cutoff_col, program_col="Program"):
    """Prepare training data for prediction"""
    df = df.copy()
    if program_col not in df.columns or cutoff_col not in df.columns:
        return None, None
        
    df["Program_norm"] = df[program_col].apply(normalize)
    prog_norm = normalize(program)
    prog_df = df[df["Program_norm"].str.contains(prog_norm, na=False)].copy()
    
    if prog_df.empty:
        return None, None

    if year_col and year_col in prog_df.columns:
        prog_df.loc[:, 'Year_Num'] = prog_df[year_col].apply(extract_year)
        prog_df = prog_df.dropna(subset=['Year_Num', cutoff_col])
        if prog_df.shape[0] >= 2:
            X = prog_df[['Year_Num']].astype(int).values
            y = prog_df[cutoff_col].astype(float).values
            return X, y
        elif prog_df.shape[0] == 1:
            # Return single data point for special handling
            return prog_df[['Year_Num']].astype(int).values, prog_df[cutoff_col].astype(float).values

    prog_df = prog_df.dropna(subset=[cutoff_col])
    if prog_df.shape[0] >= 2:
        X = np.arange(len(prog_df)).reshape(-1, 1)
        y = prog_df[cutoff_col].astype(float).values
        return X, y
    elif prog_df.shape[0] == 1:
        # Return single data point for special handling
        X = np.array([[0]])  # Use index 0 for single point
        y = prog_df[cutoff_col].astype(float).values
        return X, y

    return None, None

def get_admission_chance(user_agg, predicted_cutoff):
    """Calculate admission chance category"""
    if predicted_cutoff is None:
        return "Unknown"
    
    if user_agg >= predicted_cutoff * 1.1:
        return "High (90%+)"
    elif user_agg >= predicted_cutoff:
        return "Good (70-90%)"
    elif user_agg >= predicted_cutoff * 0.9:
        return "Possible (30-70%)"
    else:
        return "Low (<30%)"

# ========== UNIVERSITY CONFIG ==========
UNIVERSITIES = {
    "iqra": {
        "name": "Iqra University",
        "weights": {"matric": 0.10, "fsc": 0.40, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "iqra_uni_merit_list.xlsx",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Merit Percentage"
    },
    "comsats": {
        "name": "COMSATS University",
        "weights": {"matric": 0.10, "fsc": 0.40, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "comsats_merit_data.csv",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Closing Merit (%)"
    },
    "nust": {
        "name": "NUST",
        "weights": {"matric": 0.10, "fsc": 0.15, "test": 0.75},
        "totals": {"matric": 1100, "fsc": 1100, "test": 200},
        "test_used": "NET",
        "data_file": "nust_merit_data.csv",
        "program_col": "Program",
        "year_col": None,
        "cutoff_col": "Closing (%)"
    },
    "fast": {
        "name": "FAST University",
        "weights": {"matric": 0.25, "fsc": 0.25, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "fast_merit_data.csv",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Merit Score"
    },
    "bahria": {
        "name": "Bahria University",
        "weights": {"matric": 0.20, "fsc": 0.30, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "bahria_uni_merit_list.csv",
        "program_col": "Discipline",
        "year_col": "Year",
        "cutoff_col": "Aggregate"
    },
    "uet": {
        "name": "University of Engineering and Technology",
        "weights": {"fsc": 0.70, "test": 0.30},  # No matric weight for UET
        "totals": {"fsc": 1100, "test": 400},    # ECAT total marks is 400
        "test_used": "ECAT",
        "data_file": "uet_merit_list.xlsx",
        "program_col": "Field",
        "year_col": "Year",
        "cutoff_col": "Merit Score"
    },
    "ned": {
        "name": "NED University",
        "weights": {"fsc": 0.40, "test": 0.60},  # 40% HSC + 60% Entry Test
        "totals": {"fsc": 1100, "test": 100},    # Assuming test is out of 100
        "test_used": "NED Entry Test",
        "data_file": "ned_merit_list.xlsx",
        "program_col": "Discipline",
        "year_col": None,
        "cutoff_col": None  # We'll use the percentage values directly
    },
    "iiui": {
        "name": "International Islamic University Islamabad",
        "weights": {"matric": 0.40, "fsc": 0.60},  # 40% Matric + 60% FSC
        "totals": {"matric": 1100, "fsc": 1100},
        "test_used": None,  # No test required for IIUI
        "data_file": "iiui_merit_data.csv",
        "program_col": "Discipline",
        "year_col": "Year",
        "cutoff_col": "Aggregate"
    }
    
}

# ========== ROUTES ==========
@app.route("/predict", methods=["POST"])
def predict_admission():
    data = request.get_json()

    print("----------------------------------------------\n\n\n\n\n")

    print("Received data:", data)


    print("----------------------------------------------")
    
    # Validate input
    required_fields = ["matric_marks", "fsc_marks", "program"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        matric = float(data["matric_marks"])
        fsc = float(data["fsc_marks"])
        nts = float(data.get("nts_marks", 0))  # Optional
        net = float(data.get("net_marks", 0))  # Optional
        ned_test = float(data.get("ned_test_marks")) if "ned_test_marks" in data else None
        ecat = float(data.get("ecat_marks")) if "ecat_marks" in data else None  # Optional for UET
        program = data["program"]
        is_o_a_level = data.get("is_o_a_level", False)  # Detect O/A level flag
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input values"}), 400

    target_year = 2026
    results = {"universities": []}

    for uni_id, uni in UNIVERSITIES.items():
        # Copy config to avoid modifying original
        uni_config = uni.copy()
        uni_config["totals"] = uni["totals"].copy()
        
        # Override totals if O/A level
        if is_o_a_level:
            if "matric" in uni_config["totals"]:
                uni_config["totals"]["matric"] = 900  # O-level equiv max
            # fsc remains 1100 for A-level equiv
        
        # Skip if test is required but not provided
        if uni_config["test_used"] == "ECAT" and ecat is None:
            uni_result = {
                "id": uni_id,
                "name": uni_config["name"],
                "user_aggregate": None,
                "predicted_2026_cutoff": None,
                "admitted": None,
                "admission_chance": "ECAT score required",
                "criteria": uni_config,
                "last_actual_cutoff": None,
                "last_actual_year": None
            }
            results["universities"].append(uni_result)
            continue
            
        if uni_config["test_used"] == "NED Entry Test" and ned_test is None:
            uni_result = {
                "id": uni_id,
                "name": uni_config["name"],
                "user_aggregate": None,
                "predicted_2026_cutoff": None,
                "admitted": None,
                "admission_chance": "NED Entry Test score required",
                "criteria": uni_config,
                "last_actual_cutoff": None,
                "last_actual_year": None
            }
            results["universities"].append(uni_result)
            continue
            
        # Calculate user aggregate
        test_marks = None
        if uni_config["test_used"] == "NTS":
            test_marks = nts
        elif uni_config["test_used"] == "NET":
            test_marks = net
        elif uni_config["test_used"] == "ECAT":
            test_marks = ecat
        elif uni_config["test_used"] == "NED Entry Test":
            test_marks = ned_test

        user_agg = calculate_aggregate(
            matric, fsc, test_marks,
            uni_config["totals"], 
            uni_config["weights"]
        )
        
        # Try to load data and predict cutoff
        try:
            if uni_id == "ned":
                # For NED, we'll use the image data provided
                ned_data = {
                    "Discipline": ["Software Engineering (SE)", "Computer Systems Engineer", 
                                 "Computer Science and Info", "Data Sciences (DS)", 
                                 "Artificial Intelligence (AI)"],
                    "2024": [86.86, 83.90, 84.27, None, None],
                    "2023": [86.86, 83.90, 84.27, 83.73, 83.50],
                    "2022": [91.50, 89.18, 89.45, 88.40, 88.14],
                    "2021": [83.35, 83.61, 83.59, None, None],
                    "2020": [91.76, 89.24, 89.37, 83.91, 83.60],
                    "2019": [92.18, 89.11, 89.09, 84.01, 82.82],
                    "2018": [92.56, 89.02, 90.08, 83.48, 83.32]
                }
                df = pd.DataFrame(ned_data)
                df = df.melt(id_vars=["Discipline"], var_name="Year", value_name="Percentage")
                df['Year'] = df['Year'].astype(int)
            elif uni_id == "iiui":
                # For IIUI, we'll use the data from the image provided and add some synthetic historical data
                iiui_data = {
                    "Discipline": [
                        "BS Computer Science", 
                        "BS Software Engineering", 
                        "BS Artificial Intelligence", 
                        "BS Data Science", 
                        "BS Cyber Security", 
                        "BS Information Technology", 
                        "BE Electrical Engineering", 
                        "BE Mechanical Engineer",
                        # Add some synthetic historical data for prediction
                        "BS Computer Science", 
                        "BS Software Engineering", 
                        "BS Artificial Intelligence", 
                    ],
                    "Year": [2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 
                             2020, 2020, 2020],  # Added synthetic 2020 data
                    "Aggregate": [84.20, 84.10, 84.00, 83.90, 83.80, 83.70, 83.60, 83.50,
                                  83.70, 83.60, 83.50]  # Slightly lower for 2020
                }
                df = pd.DataFrame(iiui_data)
            else:
                df = pd.read_excel(uni_config["data_file"]) if str(uni_config["data_file"]).lower().endswith((".xlsx", ".xls")) else pd.read_csv(uni_config["data_file"])
        except Exception as e:
            print(f"Error loading {uni_config['data_file']}: {e}")
            predicted_cutoff = None
            latest_cutoff = None
            latest_year = None
        else:
            # Get latest actual cutoff
            latest_cutoff, latest_year = get_latest_cutoff(
                df, program, 
                "Percentage" if uni_id == "ned" else uni_config["cutoff_col"], 
                "Year" if uni_id == "ned" else uni_config["year_col"],
                "Discipline" if uni_id in ["ned", "iiui"] else uni_config.get("program_col", "Program")
            )
            
            # Prepare training data and predict
            Xy = prepare_training_data(
                df, program, 
                "Year" if uni_id in ["ned", "iiui"] else uni_config["year_col"], 
                "Percentage" if uni_id == "ned" else uni_config["cutoff_col"],
                "Discipline" if uni_id in ["ned", "iiui"] else uni_config.get("program_col", "Program")
            )
            
            if Xy[0] is not None:
                X, y = Xy
                if len(X) == 1:
                    # Handle single data point case
                    if X[0][0] > 1900:  # If it's a year value
                        current_year = X[0][0]
                    else:
                        current_year = 2021  # Default to 2021 for IIUI
                    
                    # Use a simple prediction method for single data point
                    predicted_cutoff = predict_with_single_point(
                        y[0], target_year, current_year, trend="increasing"
                    )
                else:
                    predicted_cutoff = predict_cutoff(X, y, target_year)
            else:
                predicted_cutoff = None
        
        # Determine admission status and chance
        admitted = bool(user_agg >= predicted_cutoff) if (predicted_cutoff is not None and user_agg is not None) else None
        chance = get_admission_chance(user_agg, predicted_cutoff) if predicted_cutoff is not None else "Unknown"
        
        # Prepare university result
        uni_result = {
            "id": uni_id,
            "name": uni_config["name"],
            "user_aggregate": round(user_agg, 2) if user_agg is not None else None,
            "predicted_2026_cutoff": round(predicted_cutoff, 2) if predicted_cutoff is not None else None,
            "admitted": admitted,
            "admission_chance": chance,
            "criteria": {
                "weights": uni_config["weights"],
                "totals": uni_config["totals"],
                "test_used": uni_config["test_used"]
            },
            "last_actual_cutoff": latest_cutoff,
            "last_actual_year": latest_year
        }
        
        results["universities"].append(uni_result)

    print("----------------------------------------------\n\n\n\n\n")

    print("Final results:", results)
    

    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)