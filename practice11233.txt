from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import requests
from bs4 import BeautifulSoup
import datetime
import os
import json

app = Flask(__name__)
CORS(app)  

json_path = r"C:\work\unis_recommendation\all_uni.json"

def load_cached_data():
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return {}

def save_cached_data(data):
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving cache: {e}")

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


def safe_get(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response, None
    except Exception as e:
        return None, str(e)


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


def scrape_iiui_fees():
    cache_key = "iiui_fees"
    url = UNIVERSITIES["iiui"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.content, 'html.parser')
        tables = soup.find_all('table', class_=lambda x: x and 'table' in x)
        fee_data = []
        for table in tables:
            header_row = table.find('tr')
            if not header_row:
                continue
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            if len(headers) >= 2 and any(word in headers[0].lower() for word in ['program', 'name']):
                for row in table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        program = cols[0].get_text(strip=True)
                        duration = cols[1].get_text(strip=True)
                        fee_text = cols[2].get_text(strip=True)
                        fee = None
                        if 'fee' in fee_text.lower():
                            numbers = [int(s) for s in fee_text.split() if s.isdigit()]
                            if numbers:
                                fee = numbers[-1]
                        fee_data.append({
                            'program': program,
                            'duration': duration,
                            'total_fee': f"PKR {fee}" if fee else "Not available",
                            'currency': 'PKR' if fee else None
                        })
        if fee_data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": fee_data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return fee_data, None, update_time, False
        else:
            error = "No fee data found in any tables on the IIUI page"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found in any tables on the IIUI page", None, False

@app.route('/feesiiui', methods=['GET'])
def fees_iiui():
    data, error, last_updated, from_cache = scrape_iiui_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "International Islamic University Islamabad (IIUI)",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "iiui",
        "source": "International Islamic University Islamabad (IIUI)",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_uet_fees():
    cache_key = "uet_fees"
    url = UNIVERSITIES["uet"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find("table")
        if table:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cols:
                    rows.append(dict(zip(headers, cols)))
            if rows:
                update_time = datetime.datetime.now().isoformat()
                cached[cache_key] = {
                    "fee_structure": rows,
                    "last_updated": update_time
                }
                save_cached_data(cached)
                return rows, None, update_time, False
            else:
                error = "No fee data found in the table"
        else:
            error = "Fee structure table not found"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "Fee structure table not found", None, False

@app.route('/feesuet', methods=['GET'])
def fees_uet():
    data, error, last_updated, from_cache = scrape_uet_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "University of Engineering and Technology (UET) Lahore",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "uet",
        "source": "University of Engineering and Technology (UET) Lahore",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_lums_fees():
    cache_key = "lums_fees"
    url = UNIVERSITIES["lums"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, "html.parser")
        data = {}
        bscs_table = soup.find("h2", text=lambda t: t and "BSCS" in t)
        if bscs_table:
            table = bscs_table.find_next("table")
            if table:
                headers = [th.get_text(strip=True) for th in table.find_all("th")]
                rows = []
                for tr in table.find_all("tr")[1:]:
                    cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if cols:
                        rows.append(dict(zip(headers, cols)))
                data["freshman_bscs"] = rows

        for section in ["Ph. D Programs", "Masters Programs", "Bachelors Programs", "M. Phil Programs", "Others Programs"]:
            heading = soup.find(lambda tag: tag.name.startswith("h") and section in tag.get_text())
            if heading:
                table = heading.find_next("table")
                if table:
                    rows = []
                    for tr in table.find_all("tr")[1:]:
                        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                        if len(cols) >= 3:
                            rows.append({
                                "program": cols[0],
                                "duration_years": cols[1],
                                "fee": cols[2]
                            })
                    data[section.lower().replace(" ", "_")] = rows

        if data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return data, None, update_time, False
        else:
            error = "No fee data found for LUMS"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found for LUMS", None, False

@app.route("/feeslums", methods=["GET"])
def fees_lums():
    data, error, last_updated, from_cache = scrape_lums_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "Lahore University of Management Sciences (LUMS)",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "lums",
        "source": "Lahore University of Management Sciences (LUMS)",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_ned_fees():
    cache_key = "ned_fees"
    url = UNIVERSITIES["ned"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        fee_data = []
        for table in tables:
            headers = [th.text.strip() for th in table.find_all('th')]
            for tr in table.find_all('tr')[1:]:
                cols = [td.text.strip() for td in tr.find_all('td')]
                if cols:
                    fee_data.append(dict(zip(headers, cols)))
        if fee_data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": fee_data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return fee_data, None, update_time, False
        else:
            error = "No fee data found for NED University"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found for NED University", None, False

@app.route('/nedfees', methods=['GET'])
def ned_fees():
    data, error, last_updated, from_cache = scrape_ned_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "NED University",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "ned",
        "source": "NED University",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def extract_fee_structure_air(soup, section_title):
    section = soup.find('h2', string=lambda text: text and section_title.lower() in text.lower())
    if section:
        table = section.find_next('table')
        if table:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = []
            for tr in table.find_all('tr')[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all('td')]
                if cols:
                    rows.append(dict(zip(headers, cols)))
            return rows
    return []

def scrape_fee_structure_air():
    cache_key = "air_fees"
    url = UNIVERSITIES["air"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        data = {
            'BSCS': extract_fee_structure_air(soup, "BSCS Fee Structure"),
            'BBA': extract_fee_structure_air(soup, "BBA Fee Structure"),
            'MPhil Programs': extract_fee_structure_air(soup, "M. Phil Programs"),
            'Masters Programs': extract_fee_structure_air(soup, "Masters Programs")
        }
        if any(data.values()):
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return data, None, update_time, False
        else:
            error = "No fee data found for AIR"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found for air", None, False

@app.route('/feesair', methods=['GET'])
def fees_air():
    data, error, last_updated, from_cache = scrape_fee_structure_air()
    if error:
        return jsonify({
            "status": "error",
            "source": "AIR UNIVERSITY (AIR)",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "air",
        "source": "AIR University (air)",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_nust_fees():
    cache_key = "nust_fees"
    url = UNIVERSITIES["nust"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        fee_data = []
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) == len(headers):
                    fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
        if fee_data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": fee_data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return fee_data, None, update_time, False
        else:
            error = "No fee data found for NUST"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found for NUST", None, False

@app.route('/feesnust', methods=['GET'])
def fees_nust():
    data, error, last_updated, from_cache = scrape_nust_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "National University of Sciences and Technology (NUST)",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "nust",
        "source": "National University of Sciences and Technology (NUST)",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_comsats_fees():
    cache_key = "comsats_fees"
    url = UNIVERSITIES["comsats"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        fee_data = []
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) == len(headers):
                    fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
        if fee_data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": fee_data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return fee_data, None, update_time, False
        else:
            error = "No fee data found for COMSATS"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found for COMSATS", None, False

@app.route('/feescomsats', methods=['GET'])
def fees_comsats():
    data, error, last_updated, from_cache = scrape_comsats_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "COMSATS University Islamabad (Lahore Campus)",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "comsats",
        "source": "COMSATS University Islamabad (Lahore Campus)",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_uni_of_educ_fees():
    cache_key = "uni_of_education_fees"
    url = UNIVERSITIES["uni_of_education"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        fee_data = []
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) == len(headers):
                    fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
        if fee_data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": fee_data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return fee_data, None, update_time, False
        else:
            error = "No fee data found for  University of education"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    return None, error or "No fee data found for University of Education", None, False

@app.route('/fees_uni_of_education', methods=['GET'])
def fees_uni_of_education():
    data, error, last_updated, from_cache = scrape_uni_of_educ_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "University Of education",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "Univeristy of education",
        "source": "Univeristy of education",
        "fee_structure": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

def scrape_fast_fees():
    cache_key = "fast_fees"
    url = UNIVERSITIES["fast"]["fee_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find the main content area that contains fee information
        fee_content = soup.find('div', class_=['content', 'main-content']) or soup.find('main') or soup
        
        fee_data = {}
        
        # Extract tuition fee table (main focus)
        tuition_table = fee_content.find('table')
        if tuition_table:
            headers = [th.get_text(strip=True) for th in tuition_table.find_all('th')]
            tuition_data = []
            for row in tuition_table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) == len(headers):
                    tuition_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
            if tuition_data:
                fee_data["tuition_fees"] = tuition_data
        
        # Extract student activities fund
        activities_text = fee_content.find(string=re.compile(r'Student Activities Fund.*Rs[\s\d,]+'))
        if activities_text:
            activities_match = re.search(r'Student Activities Fund.*?Rs[\s]*([\d,]+)', str(activities_text))
            if activities_match:
                fee_data["student_activities_fund"] = f"Rs {activities_match.group(1)} per semester"
        
        # Extract miscellaneous fees
        misc_fees = {}
        misc_section = fee_content.find(string=re.compile(r'Miscellaneous Fees'))
        if misc_section:
            misc_parent = misc_section.find_parent()
            if misc_parent:
                # Look for list items or table rows after "Miscellaneous Fees"
                next_elem = misc_parent.find_next()
                while next_elem and not next_elem.find('table'):
                    if next_elem.name in ['p', 'div', 'li']:
                        text = next_elem.get_text(strip=True)
                        if text and 'Rs' in text and not any(admission_word in text.lower() for admission_word in ['admission', 'applicant']):
                            # Extract fee name and amount
                            fee_match = re.search(r'([^Rs]+)\s+Rs[\s\.]*([\d,]+)', text)
                            if fee_match:
                                fee_name = fee_match.group(1).strip()
                                fee_amount = f"Rs {fee_match.group(2)}"
                                if 'admission' not in fee_name.lower():
                                    misc_fees[fee_name] = fee_amount
                    next_elem = next_elem.find_next()
        
        # Also check if there's a table for miscellaneous fees
        misc_tables = fee_content.find_all('table')
        if len(misc_tables) > 1:  # If there are multiple tables, check the second one for misc fees
            misc_table = misc_tables[1]
            headers = [th.get_text(strip=True) for th in misc_table.find_all('th')]
            for row in misc_table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    fee_name = cols[0].get_text(strip=True)
                    fee_amount = cols[1].get_text(strip=True)
                    if fee_name and 'Rs' in fee_amount and 'admission' not in fee_name.lower():
                        misc_fees[fee_name] = fee_amount
        
        if misc_fees:
            fee_data["miscellaneous_fees"] = misc_fees
        
        # Extract late payment fine information
        late_payment_text = fee_content.find(string=re.compile(r'Late Payment Fine'))
        if late_payment_text:
            late_parent = late_payment_text.find_parent()
            if late_parent:
                late_info = late_parent.find_next('p')
                if late_info:
                    fee_data["late_payment_fine"] = late_info.get_text(strip=True)
        
        # Extract refund policy (excluding admission fee refund info)
        refund_data = {}
        refund_section = fee_content.find(string=re.compile(r'Fee Refund Policy'))
        if refund_section:
            refund_parent = refund_section.find_parent()
            if refund_parent:
                # Look for refund timeline table
                refund_table = refund_parent.find_next('table')
                if refund_table:
                    headers = [th.get_text(strip=True) for th in refund_table.find_all('th')]
                    refund_timeline = []
                    for row in refund_table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) == len(headers):
                            refund_timeline.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
                    if refund_timeline:
                        refund_data["refund_timeline"] = refund_timeline
        
        if refund_data:
            fee_data["refund_policy"] = refund_data
        
        # Extract payment methods (excluding admission-related payment info)
        payment_methods = []
        payment_section = fee_content.find(string=re.compile(r'Payment Instructions'))
        if payment_section:
            payment_parent = payment_section.find_parent()
            if payment_parent:
                next_elem = payment_parent.find_next()
                while next_elem and next_elem.name in ['p', 'li', 'div']:
                    text = next_elem.get_text(strip=True)
                    if text and not any(admission_word in text.lower() for admission_word in ['admission', 'applicant']):
                        payment_methods.append(text)
                    next_elem = next_elem.find_next()
        
        if payment_methods:
            fee_data["payment_methods"] = payment_methods[:5]  # Limit to first 5 methods
        
        if fee_data:
            update_time = datetime.datetime.now().isoformat()
            cached[cache_key] = {
                "fee_structure": fee_data,
                "last_updated": update_time
            }
            save_cached_data(cached)
            return fee_data, None, update_time, False
        else:
            error = "No fee data found for FAST University"
    
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["fee_structure"], None, cached_entry["last_updated"], True
    
    return None, error or "No fee data found for FAST University", None, False

@app.route('/feesfast', methods=['GET'])
def fees_fast():
    data, error, last_updated, from_cache = scrape_fast_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "FAST University",
            "message": error
        }), 500
    
    response = {
        "status": "success",
        "uni_id": "fast",
        "source": "FAST University",
        "fee_structure": data,
        "last_updated": last_updated
    }
    
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    
    return jsonify(response)

def scrape_nust_scholarships():
    cache_key = "nust_scholarships"
    url = UNIVERSITIES["nust"]["scholarship_url"]
    cached = load_cached_data()
    resp, error = safe_get(url)
    if not error:
        soup = BeautifulSoup(resp.text, 'html.parser')
        section = soup.find('h2', string=lambda t: t and 'nust scholarships' in t.lower())
        if section:
            text = []
            node = section.find_next_sibling()
            while node and node.name not in ['h2', 'h1']:
                if node.name == 'p':
                    text.append(node.get_text(strip=True))
                if node.name == 'ul':
                    for li in node.find_all('li'):
                        text.append(f"- {li.get_text(strip=True)}")
                node = node.find_next_sibling()
            if text:
                update_time = datetime.datetime.now().isoformat()
                cached[cache_key] = {
                    "scholarships": text,
                    "last_updated": update_time
                }
                save_cached_data(cached)
                return text, None, update_time, False
            else:
                error = "No scholarship data found for NUST"
        else:
            error = "NUST scholarships section not found"
    if cache_key in cached:
        cached_entry = cached[cache_key]
        return cached_entry["scholarships"], None, cached_entry["last_updated"], True
    return None, error or "NUST scholarships section not found", None, False

@app.route('/scholarshipsnust', methods=['GET'])
def scholarships_nust():
    data, error, last_updated, from_cache = scrape_nust_scholarships()
    if error:
        return jsonify({
            "status": "error",
            "source": "National University of Sciences and Technology (NUST) Scholarships",
            "message": error
        }), 500
    response = {
        "status": "success",
        "uni_id": "nust",
        "source": "National University of Sciences and Technology (NUST) Scholarships",
        "scholarships": data,
        "last_updated": last_updated
    }
    if from_cache:
        response["note"] = "Data loaded from cache due to fetch failure"
    return jsonify(response)

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
            'NUST Scholarships': '/scholarshipsnust'
        },
        'note': 'All endpoints return JSON data. Use /predict for admission predictions and other endpoints for fee structures or scholarships.'
    })
SISGP_URL = "https://www.ilmkidunya.com/scholarships/sisgp-scholarships"
@app.route("/sisgp", methods=["GET"])
def scrape_sisgp():
    try:
        response = requests.get(SISGP_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        # Introduction - first few paragraphs after title
        intro_paragraphs = []
        intro_p = soup.find("p")
        if intro_p:
            intro_paragraphs = [p.get_text(strip=True) for p in intro_p.find_all_next("p", limit=3) if len(p.get_text(strip=True)) > 50]

        # Why Pakistani Professionals Should Apply
        why_apply_points = []
        why_section = soup.find("h2", string=lambda t: t and "Why Pakistani Professionals Should Apply" in t)
        if why_section:
            ul = why_section.find_next("ul")
            if ul:
                why_apply_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Eligibility Criteria
        eligibility_points = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Eligibility Criteria" in t)
        if eligibility_section:
            ul = eligibility_section.find_next("ul")
            if ul:
                eligibility_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        # What Does the Scholarship Cover?
        coverage_points = []
        coverage_section = soup.find("h2", string=lambda t: t and "What Does the Scholarship Cover?" in t)
        if coverage_section:
            ul = coverage_section.find_next("ul")
            if ul:
                coverage_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        # How to Apply
        apply_steps = []
        apply_section = soup.find("h2", string=lambda t: t and "How to Apply" in t)
        if apply_section:
            ol = apply_section.find_next("ol")
            if ol:
                apply_steps = [li.get_text(strip=True) for li in ol.find_all("li")]
            else:
                # Fallback to paragraphs
                steps = apply_section.find_all_next("p", limit=10)
                apply_steps = [s.get_text(strip=True) for s in steps if len(s.get_text(strip=True)) > 30 and any(num in s.get_text() for num in ['1.', '2.', '3.', '4.'])]

        # Key Dates
        dates = []
        dates_section = soup.find("h2", string=lambda t: t and "Key Dates" in t)
        if dates_section:
            ul = dates_section.find_next("ul")
            if ul:
                dates = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Benefits
        benefits_points = []
        benefits_section = soup.find("h2", string=lambda t: t and "Benefits" in t)
        if benefits_section:
            ul = benefits_section.find_next("ul")
            if ul:
                benefits_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Final Thoughts
        final_thoughts = []
        final_section = soup.find("h2", string=lambda t: t and "Final Thoughts" in t)
        if final_section:
            final_paragraphs = final_section.find_all_next("p", limit=3)
            final_thoughts = [p.get_text(strip=True) for p in final_paragraphs if len(p.get_text(strip=True)) > 30]

        # Social Media Sharing Buttons
        sharing_buttons = []
        sharing_section = soup.find_all("button", string=lambda t: t and any(platform in t.lower() for platform in ["whatsapp", "facebook", "twitter", "linkedin", "pinterest", "email"]))
        if sharing_section:
            sharing_buttons = [btn.get_text(strip=True).lower() for btn in sharing_section]

        # Feedback Options
        feedback_options = []
        feedback_section = soup.find("div", string=lambda t: t and "Is this page helpful?" in t)
        if feedback_section:
            feedback_buttons = feedback_section.find_all_next("button", limit=2)
            feedback_options = [btn.get_text(strip=True) for btn in feedback_buttons if btn.get_text(strip=True) in ["Yes", "No"]]

        # Community and Subscription Info
        community_info = []
        community_sections = soup.find_all("p", string=lambda t: t and any(keyword in t.lower() for keyword in ["followers", "subscribers", "join", "follow us"]))
        for section in community_sections:
            text = section.get_text(strip=True)
            if len(text) > 20:
                community_info.append(text)

        data = {
            "title": title_text,
            "introduction": intro_paragraphs,
            "why_apply": why_apply_points,
            "eligibility": eligibility_points,
            "coverage": coverage_points,
            "apply_steps": apply_steps,
            "key_dates": dates,
            "benefits": benefits_points,
            "final_thoughts": final_thoughts,
            "sharing_buttons": sharing_buttons,
            "feedback_options": feedback_options,
            "community_info": community_info
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Türkiye Burslari Scraper
# -----------------------
TURKIYE_URL = "https://www.ilmkidunya.com/scholarships/turkiye-burslari-scholarships"

@app.route("/turkiye", methods=["GET"])
def scrape_turkiye():
    try:
        response = requests.get(TURKIYE_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        # Introduction - first paragraph after title
        intro_p = soup.find("p")
        intro_text = intro_p.get_text(strip=True) if intro_p else "No Intro Found"

        # Why Game-Changer - section
        why_section = soup.find("h2", string=lambda t: t and "Why Türkiye Scholarships Are a Game-Changer" in t)
        why_points = []
        if why_section:
            ul = why_section.find_next("ul")
            if ul:
                why_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Eligibility
        eligibility_academic = []
        eligibility_age = []
        eligibility_other = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Who is Eligible?" in t)
        if eligibility_section:
            # Academic
            academic_h3 = eligibility_section.find_next("h3", string=lambda t: t and "Academic Requirements" in t)
            if academic_h3:
                ul = academic_h3.find_next("ul")
                if ul:
                    eligibility_academic = [li.get_text(strip=True) for li in ul.find_all("li")]
            # Age
            age_h3 = eligibility_section.find_next("h3", string=lambda t: t and "Age Limits" in t)
            if age_h3:
                ul = age_h3.find_next("ul")
                if ul:
                    eligibility_age = [li.get_text(strip=True) for li in ul.find_all("li")]
            # Other
            other_h3 = eligibility_section.find_next("h3", string=lambda t: t and "Other Requirements" in t)
            if other_h3:
                ul = other_h3.find_next("ul")
                if ul:
                    eligibility_other = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Benefits - table
        benefits_table = soup.find("table")
        benefits = {}
        if benefits_table:
            rows = benefits_table.find_all("tr")[1:]  # skip header
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)
                    benefits[key] = value

        # How to Apply
        apply_steps = []
        apply_section = soup.find("h2", string=lambda t: t and "How to Apply" in t)
        if apply_section:
            ol = apply_section.find_next("ol")
            if ol:
                apply_steps = [li.get_text(strip=True) for li in ol.find_all("li")]
            else:
                # fallback to paragraphs
                steps = apply_section.find_all_next("p", limit=10)
                apply_steps = [s.get_text(strip=True) for s in steps if len(s.get_text(strip=True)) > 30 and any(num in s.get_text() for num in ['1.', '2.', '3.', '4.'])]

        # Selection Process
        selection_steps = []
        selection_section = soup.find("h2", string=lambda t: t and "Selection Process" in t)
        if selection_section:
            ul = selection_section.find_next("ul")
            if ul:
                selection_steps = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Key Dates
        dates = []
        dates_section = soup.find("h2", string=lambda t: t and "Key Dates" in t)
        if dates_section:
            ul = dates_section.find_next("ul")
            if ul:
                dates = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Why Study in Turkey
        why_turkey_points = []
        why_turkey_section = soup.find("h2", string=lambda t: t and "Why Study in Turkey?" in t)
        if why_turkey_section:
            ul = why_turkey_section.find_next("ul")
            if ul:
                why_turkey_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        data = {
            "title": title_text,
            "introduction": intro_text,
            "why_game_changer": why_points,
            "eligibility": {
                "academic": eligibility_academic,
                "age": eligibility_age,
                "other": eligibility_other
            },
            "benefits": benefits,
            "apply_steps": apply_steps,
            "selection_process": selection_steps,
            "key_dates": dates,
            "why_study_turkey": why_turkey_points
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Stipendium Hungaricum Scraper
# -----------------------
STIPENDIUM_URL = "https://www.ilmkidunya.com/scholarships/stipendium-hungaricum-scholarships"

@app.route("/hungary", methods=["GET"])
def scrape_stipendium():
    try:
        response = requests.get(STIPENDIUM_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        # Introduction
        intro_paragraphs = []
        intro_section = soup.find("h2", string=lambda t: t and "Introduction" in t)
        if intro_section:
            intro_paragraphs = [p.get_text(strip=True) for p in intro_section.find_all_next("p", limit=3) if len(p.get_text(strip=True)) > 50]

        # What is the Stipendium Hungaricum Scholarship?
        what_is = []
        what_section = soup.find("h2", string=lambda t: t and "What is the Stipendium Hungaricum Scholarship?" in t)
        if what_section:
            what_paragraphs = what_section.find_all_next("p", limit=3)
            what_is = [p.get_text(strip=True) for p in what_paragraphs if len(p.get_text(strip=True)) > 30]

        # Programs Offered
        programs = []
        programs_section = soup.find("h2", string=lambda t: t and "Programs Offered" in t)
        if programs_section:
            ul = programs_section.find_next("ul")
            if ul:
                programs = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Eligibility Criteria
        eligibility = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Eligibility Criteria" in t)
        if eligibility_section:
            ul = eligibility_section.find_next("ul")
            if ul:
                eligibility = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Scholarship Benefits
        benefits = []
        benefits_section = soup.find("h2", string=lambda t: t and "Scholarship Benefits" in t)
        if benefits_section:
            ul = benefits_section.find_next("ul")
            if ul:
                benefits = [li.get_text(strip=True) for li in ul.find_all("li")]

        # How to Apply
        apply_steps = []
        apply_section = soup.find("h2", string=lambda t: t and "How to Apply" in t)
        if apply_section:
            steps = apply_section.find_all_next("p", limit=10)
            apply_steps = [s.get_text(strip=True) for s in steps if len(s.get_text(strip=True)) > 30]

        # Important Dates
        dates = []
        dates_section = soup.find("h2", string=lambda t: t and "Important Dates" in t)
        if dates_section:
            ul = dates_section.find_next("ul")
            if ul:
                dates = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Why Pakistani Students Should Apply
        why_apply = []
        why_section = soup.find("h2", string=lambda t: t and "Why Pakistani Students Should Apply" in t)
        if why_section:
            ul = why_section.find_next("ul")
            if ul:
                why_apply = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Contact for Queries
        contact_info = []
        contact_section = soup.find("h2", string=lambda t: t and "Contact for Queries" in t)
        if contact_section:
            contact_paragraphs = contact_section.find_all_next("p", limit=3)
            contact_info = [p.get_text(strip=True) for p in contact_paragraphs if len(p.get_text(strip=True)) > 20]

        # Conclusion
        conclusion = []
        conclusion_section = soup.find("h2", string=lambda t: t and "Conclusion" in t)
        if conclusion_section:
            conclusion_paragraphs = conclusion_section.find_all_next("p", limit=3)
            conclusion = [p.get_text(strip=True) for p in conclusion_paragraphs if len(p.get_text(strip=True)) > 30]

        # Social Media Sharing Buttons
        sharing_buttons = []
        sharing_section = soup.find_all("button", string=lambda t: t and any(platform in t.lower() for platform in ["whatsapp", "facebook", "twitter", "linkedin", "pinterest", "email"]))
        if sharing_section:
            sharing_buttons = [btn.get_text(strip=True).lower() for btn in sharing_section]

        # Feedback Options
        feedback_options = []
        feedback_section = soup.find("div", string=lambda t: t and "Is this page helpful?" in t)
        if feedback_section:
            feedback_buttons = feedback_section.find_all_next("button", limit=2)
            feedback_options = [btn.get_text(strip=True) for btn in feedback_buttons if btn.get_text(strip=True) in ["Yes", "No"]]

        # Community and Subscription Info
        community_info = []
        community_sections = soup.find_all("p", string=lambda t: t and any(keyword in t.lower() for keyword in ["followers", "subscribers", "join", "follow us"]))
        for section in community_sections:
            text = section.get_text(strip=True)
            if len(text) > 20:
                community_info.append(text)

        # Copyright Notice
        copyright_notice = ""
        copyright_section = soup.find("p", string=lambda t: t and "Copyright" in t)
        if copyright_section:
            copyright_notice = copyright_section.get_text(strip=True)

        data = {
            "title": title_text,
            "introduction": intro_paragraphs,
            "what_is": what_is,
            "programs": programs,
            "eligibility": eligibility,
            "benefits": benefits,
            "apply_steps": apply_steps,
            "important_dates": dates,
            "why_apply": why_apply,
            "contact_info": contact_info,
            "conclusion": conclusion,
            "sharing_buttons": sharing_buttons,
            "feedback_options": feedback_options,
            "community_info": community_info,
            "copyright_notice": copyright_notice
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Chevening Scraper
# -----------------------
CHEVENING_URL = "https://www.ilmkidunya.com/scholarships/chevening-scholarships"

@app.route("/chevening", methods=["GET"])
def scrape_chevening():
    try:
        response = requests.get(CHEVENING_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        facts = {}
        quick_facts = soup.find_all("tr")
        for row in quick_facts:
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                facts[key] = value

        paragraphs = [
            p.get_text(strip=True) for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]

        data = {
            "title": title_text,
            "facts": facts,
            "paragraphs": paragraphs[:10]
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Erasmus Scraper
# -----------------------
ERASMUS_URL = "https://www.ilmkidunya.com/scholarships/erasmus-mundus-scholarships"

@app.route("/erasmus", methods=["GET"])
def scrape_erasmus():
    try:
        response = requests.get(ERASMUS_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        facts = {}
        quick_facts = soup.find_all("tr")
        for row in quick_facts:
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                facts[key] = value

        headings = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"])]

        paragraphs = [
            p.get_text(strip=True) for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]

        data = {
            "title": title_text,
            "facts": facts,
            "headings": headings,
            "paragraphs": paragraphs[:12]
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Commonwealth Scraper
# -----------------------
COMMONWEALTH_URL = "https://www.ilmkidunya.com/scholarships/commonwealth-international-scholarships"

@app.route("/commonwealth", methods=["GET"])
def scrape_commonwealth():
    try:
        response = requests.get(COMMONWEALTH_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        facts = {}
        quick_facts = soup.find_all("tr")
        for row in quick_facts:
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                facts[key] = value

        headings = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"])]

        paragraphs = [
            p.get_text(strip=True) for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 60
        ]

        data = {
            "title": title_text,
            "facts": facts,
            "headings": headings,
            "paragraphs": paragraphs[:15]
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Rhodes Scraper
# -----------------------
RHODES_URL = "https://www.ilmkidunya.com/scholarships/rhodes-uk-scholarships"

@app.route("/rhodes", methods=["GET"])
def scrape_rhodes():
    try:
        response = requests.get(RHODES_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        # Introduction
        introduction = ""
        intro_section = soup.find("h2", string=lambda t: t and "Introduction" in t)
        if intro_section:
            next_p = intro_section.find_next("p")
            if next_p:
                introduction = next_p.get_text(strip=True)

        # Quick Facts
        facts = {}
        quick_facts = soup.find_all("tr")
        for row in quick_facts:
            cols = row.find_all(["td", "th"])
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                facts[key] = value

        # What the Scholarship Covers
        benefits = []
        benefits_section = soup.find("h2", string=lambda t: t and "What the Scholarship Covers" in t)
        if benefits_section:
            ul = benefits_section.find_next("ul")
            if ul:
                benefits = [li.get_text(strip=True) for li in ul.find_all("li")]
            else:
                # Alternative approach if structured differently
                next_p = benefits_section.find_next("p")
                if next_p:
                    benefits = [next_p.get_text(strip=True)]

        # Eligibility Criteria
        eligibility = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Eligibility Criteria" in t)
        if eligibility_section:
            ul = eligibility_section.find_next("ul")
            if ul:
                eligibility = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Application Process
        application_process = []
        application_section = soup.find("h2", string=lambda t: t and "Application Process" in t)
        if application_section:
            # Get all list items under the application process section
            current = application_section.find_next()
            while current and current.name != "h2":
                if current.name == "h3" or current.name == "h4":
                    process_step = {"title": current.get_text(strip=True), "details": ""}
                    application_process.append(process_step)
                elif current.name == "p":
                    if application_process:
                        application_process[-1]["details"] += current.get_text(strip=True) + " "
                elif current.name == "ul":
                    if application_process:
                        list_items = [li.get_text(strip=True) for li in current.find_all("li")]
                        application_process[-1]["details"] += "; ".join(list_items)
                current = current.find_next_sibling()

        # Why Apply for Rhodes?
        why_apply = []
        why_apply_section = soup.find("h2", string=lambda t: t and "Why Apply for Rhodes" in t)
        if why_apply_section:
            ul = why_apply_section.find_next("ul")
            if ul:
                why_apply = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Tips for a Strong Application
        tips = []
        tips_section = soup.find("h2", string=lambda t: t and "Tips for a Strong Application" in t)
        if tips_section:
            ul = tips_section.find_next("ul")
            if ul:
                tips = [li.get_text(strip=True) for li in ul.find_all("li")]

        # Final Thoughts
        final_thoughts = ""
        final_section = soup.find("h2", string=lambda t: t and "Final Words" in t) or soup.find("h2", string=lambda t: t and "Final Thoughts" in t)
        if final_section:
            next_p = final_section.find_next("p")
            if next_p:
                final_thoughts = next_p.get_text(strip=True)

        # Important Notes
        important_notes = []
        notes_section = soup.find("strong", string=lambda t: t and "Important:" in t)
        if notes_section:
            important_notes = [notes_section.parent.get_text(strip=True)]

        data = {
            "title": title_text,
            "introduction": introduction,
            "quick_facts": facts,
            "benefits": benefits,
            "eligibility": eligibility,
            "application_process": application_process,
            "why_apply": why_apply,
            "tips": tips,
            "final_thoughts": final_thoughts,
            "important_notes": important_notes,
            "source_url": RHODES_URL
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)