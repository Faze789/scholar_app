from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ========== HELPER FUNCTIONS (Admission Prediction) ==========
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
        years_diff = target_year - current_year
        return y_value + (years_diff * 0.5)
    elif trend == "decreasing":
        years_diff = target_year - current_year
        return y_value - (years_diff * 0.3)
    else:
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

# ========== HELPER: Safe requests (Fee Scraping) ==========
def safe_get(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response, None
    except Exception as e:
        return None, str(e)

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
        "cutoff_col": "Merit Percentage",
        "fee_url": "https://www.ilmkidunya.com/colleges/iqra-university-karachi-fee-structure.aspx"
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
        "cutoff_col": "Aggregate",
        "fee_url": "https://www.ilmkidunya.com/colleges/bahria-university-karachi-fee-structure.aspx"
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
        "data_file": "iiui_merit_data.csv",
        "program_col": "Discipline",
        "year_col": "Year",
        "cutoff_col": "Aggregate",
        "fee_url": "https://www.ilmkidunya.com/colleges/international-islamic-university-islamabad-fee-structure.aspx"
    },
    "uol": {
        "name": "University of Lahore",
        "weights": {"matric": 0.20, "fsc": 0.30, "test": 0.50},
        "totals": {"matric": 1100, "fsc": 1100, "test": 100},
        "test_used": "NTS",
        "data_file": "uol_merit_data.csv",
        "program_col": "Program",
        "year_col": "Year",
        "cutoff_col": "Merit Score",
        "fee_url": "https://www.ilmkidunya.com/colleges/university-of-lahore-uol-fee-structure.aspx"
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

# ========== ADMISSION PREDICTION ROUTE ==========
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
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input values"}), 400

    target_year = 2026
    results = {"universities": []}

    for uni_id, uni in UNIVERSITIES.items():
        uni_config = uni.copy()
        uni_config["totals"] = uni["totals"].copy()
        
        if is_o_a_level:
            if "matric" in uni_config["totals"]:
                uni_config["totals"]["matric"] = 900
        
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
            elif uni_id == "uol":
                uol_data = {
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
                df = pd.DataFrame(uol_data)
            else:
                if uni_config.get("data_file"):
                    df = pd.read_excel(uni_config["data_file"]) if str(uni_config["data_file"]).lower().endswith((".xlsx", ".xls")) else pd.read_csv(uni_config["data_file"])
                else:
                    df = None
        except Exception as e:
            print(f"Error loading {uni_config.get('data_file', 'data')}: {e}")
            predicted_cutoff = None
            latest_cutoff = None
            latest_year = None
        else:
            if df is not None:
                latest_cutoff, latest_year = get_latest_cutoff(
                    df, program, 
                    "Percentage" if uni_id == "ned" else ("Merit Score" if uni_id == "uol" else uni_config["cutoff_col"]), 
                    "Year" if uni_id in ["ned", "iiui", "uol"] else uni_config["year_col"],
                    "Discipline" if uni_id in ["ned", "iiui"] else ("Program" if uni_id == "uol" else uni_config.get("program_col", "Program"))
                )
                
                Xy = prepare_training_data(
                    df, program, 
                    "Year" if uni_id in ["ned", "iiui", "uol"] else uni_config["year_col"], 
                    "Percentage" if uni_id == "ned" else ("Merit Score" if uni_id == "uol" else uni_config["cutoff_col"]),
                    "Discipline" if uni_id in ["ned", "iiui"] else ("Program" if uni_id == "uol" else uni_config.get("program_col", "Program"))
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
                    else:
                        predicted_cutoff = predict_cutoff(X, y, target_year)
                else:
                    predicted_cutoff = None
            else:
                predicted_cutoff = None
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
            "last_actual_year": latest_year
        }
        
        results["universities"].append(uni_result)

    print("Final results:", results)
    return jsonify(results)

# ========== FEE SCRAPING ROUTES ==========
def scrape_iiui_fees():
    resp, error = safe_get(UNIVERSITIES["iiui"]["fee_url"])
    if error:
        return None, f"Failed to fetch IIUI page: {error}"
    
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
    
    if not fee_data:
        return None, "No fee data found in any tables on the IIUI page"
    return fee_data, None

@app.route('/feesiiui', methods=['GET'])
def fees_iiui():
    data, error = scrape_iiui_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "International Islamic University Islamabad (IIUI)",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "International Islamic University Islamabad (IIUI)",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_uet_fees():
    resp, error = safe_get(UNIVERSITIES["uet"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find("table")
    if not table:
        return None, "Fee structure table not found"
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            rows.append(dict(zip(headers, cols)))
    return rows, None

@app.route('/feesuet', methods=['GET'])
def fees_uet():
    data, error = scrape_uet_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "University of Engineering and Technology (UET) Lahore",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "University of Engineering and Technology (UET) Lahore",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_lums_fees():
    resp, error = safe_get(UNIVERSITIES["lums"]["fee_url"])
    if error:
        return None, error
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

    if not data:
        return None, "No fee data found for LUMS"
    return data, None

@app.route("/feeslums", methods=["GET"])
def fees_lums():
    data, error = scrape_lums_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "Lahore University of Management Sciences (LUMS)",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "Lahore University of Management Sciences (LUMS)",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_ned_fees():
    resp, error = safe_get(UNIVERSITIES["ned"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    fee_data = []
    for table in tables:
        headers = [th.text.strip() for th in table.find_all('th')]
        for tr in table.find_all('tr')[1:]:
            cols = [td.text.strip() for td in tr.find_all('td')]
            if cols:
                fee_data.append(dict(zip(headers, cols)))
    return fee_data, None

@app.route('/nedfees', methods=['GET'])
def ned_fees():
    data, error = scrape_ned_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "NED University",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "NED University",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def extract_fee_structure_uol(soup, section_title):
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

def scrape_fee_structure_uol():
    resp, error = safe_get(UNIVERSITIES["uol"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    data = {
        'BSCS': extract_fee_structure_uol(soup, "BSCS Fee Structure"),
        'BBA': extract_fee_structure_uol(soup, "BBA Fee Structure"),
        'MPhil Programs': extract_fee_structure_uol(soup, "M. Phil Programs"),
        'Masters Programs': extract_fee_structure_uol(soup, "Masters Programs")
    }
    if not any(data.values()):
        return None, "No fee data found for UOL"
    return data, None

@app.route('/feesuol', methods=['GET'])
def fees_uol():
    data, error = scrape_fee_structure_uol()
    if error:
        return jsonify({
            "status": "error",
            "source": "University of Lahore (UOL)",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "University of Lahore (UOL)",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_nust_fees():
    resp, error = safe_get(UNIVERSITIES["nust"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    fee_data = []
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) == len(headers):
                fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
    if not fee_data:
        return None, "No fee data found for NUST"
    return fee_data, None

@app.route('/feesnust', methods=['GET'])
def fees_nust():
    data, error = scrape_nust_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "National University of Sciences and Technology (NUST)",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "National University of Sciences and Technology (NUST)",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_comsats_fees():
    resp, error = safe_get(UNIVERSITIES["comsats"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    fee_data = []
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) == len(headers):
                fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
    if not fee_data:
        return None, "No fee data found for COMSATS"
    return fee_data, None

@app.route('/feescomsats', methods=['GET'])
def fees_comsats():
    data, error = scrape_comsats_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "COMSATS University Islamabad (Lahore Campus)",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "COMSATS University Islamabad (Lahore Campus)",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_bahria_fees():
    resp, error = safe_get(UNIVERSITIES["bahria"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    fee_data = []
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) == len(headers):
                fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
    if not fee_data:
        return None, "No fee data found for Bahria University"
    return fee_data, None

@app.route('/feesbahria', methods=['GET'])
def fees_bahria():
    data, error = scrape_bahria_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "Bahria University",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "Bahria University",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_iqra_fees():
    resp, error = safe_get(UNIVERSITIES["iqra"]["fee_url"])
    if error:
        return None, error
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    fee_data = []
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) == len(headers):
                fee_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
    if not fee_data:
        return None, "No fee data found for Iqra University"
    return fee_data, None

@app.route('/feesiqra', methods=['GET'])
def fees_iqra():
    data, error = scrape_iqra_fees()
    if error:
        return jsonify({
            "status": "error",
            "source": "Iqra University",
            "message": error
        }), 500
    return jsonify({
        "status": "success",
        "source": "Iqra University",
        "fee_structure": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

def scrape_nust_scholarships():
    resp, err = safe_get(UNIVERSITIES["nust"]["scholarship_url"])
    if err:
        return None, err
    soup = BeautifulSoup(resp.text, 'html.parser')
    section = soup.find('h2', string=lambda t: t and 'nust scholarships' in t.lower())
    if not section:
        return None, "NUST scholarships section not found"
    text = []
    node = section.find_next_sibling()
    while node and node.name not in ['h2', 'h1']:
        if node.name == 'p':
            text.append(node.get_text(strip=True))
        if node.name == 'ul':
            for li in node.find_all('li'):
                text.append(f"- {li.get_text(strip=True)}")
        node = node.find_next_sibling()
    if not text:
        return None, "No scholarship data found for NUST"
    return text, None

@app.route('/scholarshipsnust', methods=['GET'])
def scholarships_nust():
    data, err = scrape_nust_scholarships()
    if err:
        return jsonify({
            "status": "error",
            "source": "National University of Sciences and Technology (NUST) Scholarships",
            "message": err
        }), 500
    return jsonify({
        "status": "success",
        "source": "National University of Sciences and Technology (NUST) Scholarships",
        "scholarships": data,
        "last_updated": datetime.datetime.now().isoformat()
    })

# ========== ROOT ENDPOINT ==========
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
            'University of Lahore (UOL)': '/feesuol',
            'COMSATS': '/feescomsats',
            'Bahria University': '/feesbahria',
            'Iqra University': '/feesiqra',
            'NUST Fee Structure': '/feesnust',
            'NUST Scholarships': '/scholarshipsnust'
        },
        'note': 'All endpoints return JSON data. Use /predict for admission predictions and other endpoints for fee structures or scholarships.'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)