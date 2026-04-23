from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import datetime
import os
import json
from events import events_bp
from web_scraping import web_scraping_bp, set_universities

app = Flask(__name__)
CORS(app)  


app.register_blueprint(web_scraping_bp)
app.register_blueprint(events_bp)  

UNIVERSITIES = {
   "fast": {
        "name": "FAST University",
        "weights": {"matric": 0.25, "fsc": 0.25, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "fast_merit_data.xlsx",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Merit Percentage",
        "fee_url": "https://www.nu.edu.pk/Admissions/FeeStructure"
    },
    "comsats": {
        "name": "COMSATS University",
        "weights": {"matric": 0.10, "fsc": 0.40, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "comsats_merit_data.csv",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Closing Merit (%)",
        "fee_url": "https://www.ilmkidunya.com/colleges/comsats-university-islamabad-fee-structure.aspx"
    },
    "nust": {
        "name": "NUST",
        "weights": {"matric": 0.10, "fsc": 0.15, "test": 0.75},
        "totals": {"matric": 1100, "fsc": 1100, "test": 200},
        "test_used": "NET",
        "data_file": "nust_merit_data.csv",
        "program_col": "Program",
        "year_col": None,
        "cutoff_col": "Closing (%)",
        "fee_url": "https://www.ilmkidunya.com/colleges/national-university-of-sciences-technology-nust-islamabad-fee-structure.aspx",
        "scholarship_url": "https://www.ilmkidunya.com/colleges/national-university-of-sciences-technology-nust-islamabad.aspx"
    },
    
    "uni_of_education": {
        "name": "University of education",
        "weights": {"matric": 0.20, "fsc": 0.30, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "uni_of_education_merit_list.csv",
        "program_col": "Discipline",
        "year_col": "Year",
        "cutoff_col": "Aggregate",
        "fee_url": "https://www.ilmkidunya.com/colleges/university-of-education-bank-road-lahore-fee-structure.aspx"
    },
    "uet": {
        "name": "University of Engineering and Technology",
        "weights": {"fsc": 0.70, "test": 0.30},
        "totals": {"fsc": 1100, "test": 400},
        "test_used": "ECAT",
        "data_file": "uet_merit_list.xlsx",
        "program_col": "Field",
        "year_col": "Year",
        "cutoff_col": "Merit Score",
        "fee_url": "https://www.ilmkidunya.com/colleges/university-of-engineering-and-technology-uet-lahore-fee-structure.aspx"
    },
    "ned": {
        "name": "NED University",
        "weights": {"fsc": 0.40, "test": 0.60},
        "totals": {"fsc": 1100, "test": 100},
        "test_used": "NED Entry Test",
        "data_file": "ned_merit_list.xlsx",
        "program_col": "Discipline",
        "year_col": None,
        "cutoff_col": None,
        "fee_url": "https://www.ilmkidunya.com/colleges/ned-university-of-engineering-technology-karachi-fee-structure.aspx"
    },
    "iiui": {
        "name": "International Islamic University Islamabad",
        "weights": {"matric": 0.40, "fsc": 0.60},
        "totals": {"matric": 1100, "fsc": 1100},
        "test_used": None,
        "data_file":None,
        "program_col": "Discipline",
        "year_col": "Year",
        "cutoff_col": "Aggregate",
        "fee_url": "https://www.ilmkidunya.com/colleges/international-islamic-university-islamabad-fee-structure.aspx"
    },
    "air": {
        "name": "Air University",
        "weights": {"matric": 0.20, "fsc": 0.30, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "air_merit_data.csv",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Merit Score",
        "fee_url": "https://www.ilmkidunya.com/colleges/air-university-islamabad-fee-structure.aspx"
    },
    "lums": {
        "name": "Lahore University of Management Sciences",
        "weights": {"matric": 0.10, "fsc": 0.40, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "SAT",
        "data_file": None,
        "program_col": None,
        "year_col": None,
        "cutoff_col": None,
        "fee_url": "https://www.ilmkidunya.com/colleges/lahore-university-of-management-sciences-lahore-lums-fee-structure.aspx"
    }
}

set_universities(UNIVERSITIES)

def calculate_aggregate(matric_marks, fsc_marks, test_marks, totals, weights):
    matric_pct = (matric_marks / totals["matric"]) * 100 if totals.get("matric") else 0
    fsc_pct = (fsc_marks / totals["fsc"]) * 100 if totals.get("fsc") else 0
    test_pct = (test_marks / totals["test"]) * 100 if totals.get("test") and test_marks is not None else 0

    aggregate = (matric_pct * weights.get("matric", 0)) + \
                (fsc_pct * weights.get("fsc", 0)) + \
                (test_pct * weights.get("test", 0))
    return aggregate

def normalize(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\b(bs|bsc|bachelors|in|science|scien)\b', '', str(text).lower()).strip()

def extract_year(year_str):
    match = re.search(r'(\d{4})', str(year_str).replace(',', ''))
    return int(match.group(1)) if match else None

def predict_cutoff(X, y, target_year):
    if len(X) == 0:
        return None, None, None, None
        
    if X.mean() > 1900:  
        pred_input = target_year
    else:  
        pred_input = len(X)
    
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_pred = linear_model.predict(X)
    linear_r2 = r2_score(y, linear_pred)
    
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    poly_pred = poly_model.predict(X)
    poly_r2 = r2_score(y, poly_pred)
    
    if poly_r2 > linear_r2:
        best_model = "polynomial"
        predicted_value = float(poly_model.predict(np.array([[pred_input]]))[0])
    else:
        best_model = "linear"
        predicted_value = float(linear_model.predict(np.array([[pred_input]]))[0])
    
    return predicted_value, linear_r2, poly_r2, best_model

def predict_with_single_point(y_value, target_year, current_year, trend="stable"):
    if trend == "increasing":
        years_diff = target_year - current_year
        return y_value + (years_diff * 0.5)
    elif trend == "decreasing":
        years_diff = target_year - current_year
        return y_value - (years_diff * 0.3)
    else:
        return y_value

def get_latest_cutoff(df, program, cutoff_col, year_col=None, program_col="Program"):
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
            return prog_df[['Year_Num']].astype(int).values, prog_df[cutoff_col].astype(float).values

    prog_df = prog_df.dropna(subset=[cutoff_col])
    if prog_df.shape[0] >= 2:
        X = np.arange(len(prog_df)).reshape(-1, 1)
        y = prog_df[cutoff_col].astype(float).values
        return X, y
    elif prog_df.shape[0] == 1:
        X = np.array([[0]])
        y = prog_df[cutoff_col].astype(float).values
        return X, y

    return None, None

def get_admission_chance(user_agg, predicted_cutoff):
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

@app.route("/predict", methods=["POST"])
def predict_admission():
    data = request.get_json()
    print("Received data:", data)
    
    required_fields = ["matric_marks", "fsc_marks", "program"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        matric = float(data["matric_marks"])
        fsc = float(data["fsc_marks"])
        nts = float(data.get("nts_marks", 0))
        net = float(data.get("net_marks", 0))
        ned_test = float(data.get("ned_test_marks")) if "ned_test_marks" in data else None
        ecat = float(data.get("ecat_marks")) if "ecat_marks" in data else None
        sat = float(data.get("sat_marks")) if "sat_marks" in data else None
        program = data["program"]
        is_o_a_level = data.get("is_o_a_level", False)
        bachelors_cgpa = data.get("bachelors_cgpa")
        masters_cgpa = data.get("masters_cgpa")
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input values"}), 400

    is_graduate_application = bachelors_cgpa is not None or masters_cgpa is not None
    
    target_year = 2026
    results = {"universities": []}
    
    if is_graduate_application:
        results["user_data"] = {
            "matric_marks": matric,
            "fsc_marks": fsc,
            "nts_marks": nts,
            "net_marks": net,
            "ned_test_marks": ned_test,
            "ecat_marks": ecat,
            "sat_marks": sat,
            "program": program,
            "is_o_a_level": is_o_a_level,
            "bachelors_cgpa": bachelors_cgpa,
            "masters_cgpa": masters_cgpa
        }

    for uni_id, uni in UNIVERSITIES.items():
        uni_config = uni.copy()
        uni_config["totals"] = uni["totals"].copy()
        
        if is_o_a_level:
            if "matric" in uni_config["totals"]:
                uni_config["totals"]["matric"] = 900
        
        if is_graduate_application:
            uni_result = {
                "id": uni_id,
                "name": uni_config["name"],
                "user_aggregate": None,
                "predicted_2026_cutoff": None,
                "admitted": None,
                "admission_chance": "Graduate application - no prediction needed",
                "criteria": {
                    "weights": uni_config["weights"],
                    "totals": uni_config["totals"],
                    "test_used": uni_config["test_used"]
                },
                "last_actual_cutoff": None,
                "last_actual_year": None
            }
            results["universities"].append(uni_result)
            continue
            
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
            
        if uni_config["test_used"] == "SAT" and sat is None:
            uni_result = {
                "id": uni_id,
                "name": uni_config["name"],
                "user_aggregate": None,
                "predicted_2026_cutoff": None,
                "admitted": None,
                "admission_chance": "SAT score required",
                "criteria": uni_config,
                "last_actual_cutoff": None,
                "last_actual_year": None
            }
            results["universities"].append(uni_result)
            continue

        test_marks = None
        if uni_config["test_used"] == "NTS":
            test_marks = nts
        elif uni_config["test_used"] == "NET":
            test_marks = net
        elif uni_config["test_used"] == "ECAT":
            test_marks = ecat
        elif uni_config["test_used"] == "NED Entry Test":
            test_marks = ned_test
        elif uni_config["test_used"] == "SAT":
            test_marks = sat

        user_agg = calculate_aggregate(
            matric, fsc, test_marks,
            uni_config["totals"], 
            uni_config["weights"]
        )
        
        try:
            if uni_id == "ned":
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
                        "BS Computer Science", 
                        "BS Software Engineering", 
                        "BS Artificial Intelligence", 
                    ],
                    "Year": [2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 
                             2020, 2020, 2020],
                    "Aggregate": [84.20, 84.10, 84.00, 83.90, 83.80, 83.70, 83.60, 83.50,
                                  83.70, 83.60, 83.50]
                }
                df = pd.DataFrame(iiui_data)
            elif uni_id == "air":
                air_data = {
                    "Program": [
                        "Computer Science", 
                        "Software Engineering", 
                        "Business Administration", 
                        "Electrical Engineering",
                        "Computer Science", 
                        "Software Engineering", 
                        "Business Administration", 
                        "Electrical Engineering",
                        "Computer Science", 
                        "Software Engineering"
                    ],
                    "Year": [2023, 2023, 2023, 2023, 
                             2022, 2022, 2022, 2022,
                             2021, 2021],
                    "Merit Score": [75.5, 74.8, 72.3, 76.2,
                                   74.8, 74.1, 71.8, 75.5,
                                   74.2, 73.9]
                }
                df = pd.DataFrame(air_data)
            else:
                if uni_config.get("data_file"):
                    df = pd.read_excel(uni_config["data_file"]) if str(uni_config["data_file"]).lower().endswith((".xlsx", ".xls")) else pd.read_csv(uni_config["data_file"])
                else:
                    df = None
        except Exception as e:
            print(f"Error loading {uni_config.get('data_file', 'data')}: {e}")
            predicted_cutoff = None
            linear_r2 = None
            poly_r2 = None
            best_model = None
            latest_cutoff = None
            latest_year = None
        else:
            if df is not None:
                latest_cutoff, latest_year = get_latest_cutoff(
                    df, program, 
                    "Percentage" if uni_id == "ned" else ("Merit Score" if uni_id == "air" else uni_config["cutoff_col"]), 
                    "Year" if uni_id in ["ned", "iiui", "air"] else uni_config["year_col"],
                    "Discipline" if uni_id in ["ned", "iiui"] else ("Program" if uni_id == "air" else uni_config.get("program_col", "Program"))
                )
                
                Xy = prepare_training_data(
                    df, program, 
                    "Year" if uni_id in ["ned", "iiui", "air"] else uni_config["year_col"], 
                    "Percentage" if uni_id == "ned" else ("Merit Score" if uni_id == "air" else uni_config["cutoff_col"]),
                    "Discipline" if uni_id in ["ned", "iiui"] else ("Program" if uni_id == "air" else uni_config.get("program_col", "Program"))
                )
                
                if Xy[0] is not None:
                    X, y = Xy
                    if len(X) == 1:
                        if X[0][0] > 1900:
                            current_year = X[0][0]
                        else:
                            current_year = 2021
                        predicted_cutoff = predict_with_single_point(
                            y[0], target_year, current_year, trend="increasing"
                        )
                        linear_r2 = None
                        poly_r2 = None
                        best_model = "single_point"
                    else:
                        predicted_cutoff, linear_r2, poly_r2, best_model = predict_cutoff(X, y, target_year)
                else:
                    predicted_cutoff = None
                    linear_r2 = None
                    poly_r2 = None
                    best_model = None
            else:
                predicted_cutoff = None
                linear_r2 = None
                poly_r2 = None
                best_model = None
                latest_cutoff = None
                latest_year = None
        
        admitted = bool(user_agg >= predicted_cutoff) if (predicted_cutoff is not None and user_agg is not None) else None
        chance = get_admission_chance(user_agg, predicted_cutoff) if predicted_cutoff is not None else "Unknown"
        
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
            "last_actual_year": latest_year,
            "linear_r2": linear_r2,
            "poly_r2": poly_r2,
            "best_model": best_model
        }
        
        results["universities"].append(uni_result)

    print("Final results:", results)
    return jsonify(results)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'success',
        'endpoints': {
            'Admission Prediction': '/predict',
            'International Islamic University Islamabad (IIUI)': '/feesiiui',
            'UET Lahore': '/feesuet',
            'LUMS': '/feeslums',
            'NED University': '/nedfees',
            'AIR university (air)': '/feesair',
            'COMSATS': '/feescomsats',
            'Univeristy of education': '/fees_uni_of_education',
            'fast University': '/feesfast',
            'NUST Fee Structure': '/feesnust',
            'NUST Scholarships': '/scholarshipsnust',
            'COMSATS Events': '/api/comsats_events',
            'NEDUET Events': '/api/neduet_events', 
            'UET Taxila Events': '/api/uet_taxila_events'
        },
        'note': 'All endpoints return JSON data. Use /predict for admission predictions and other endpoints for fee structures, scholarships, or events.'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)