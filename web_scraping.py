import os
import json
import datetime
import re
import requests
from bs4 import BeautifulSoup
from flask import Blueprint
from flask import Flask, request, jsonify


web_scraping_bp = Blueprint('web_scraping', __name__)


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

def safe_get(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response, None
    except Exception as e:
        return None, str(e)


UNIVERSITIES = None

def set_universities(unis_dict):
    global UNIVERSITIES
    UNIVERSITIES = unis_dict


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

@web_scraping_bp.route('/feesiiui', methods=['GET'])
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

@web_scraping_bp.route('/feesuet', methods=['GET'])
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

@web_scraping_bp.route("/feeslums", methods=["GET"])
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

@web_scraping_bp.route('/nedfees', methods=['GET'])
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

@web_scraping_bp.route('/feesair', methods=['GET'])
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

@web_scraping_bp.route('/feesnust', methods=['GET'])
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

@web_scraping_bp.route('/feescomsats', methods=['GET'])
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

@web_scraping_bp.route('/fees_uni_of_education', methods=['GET'])
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
        
      
        fee_content = soup.find('div', class_=['content', 'main-content']) or soup.find('main') or soup
        
        fee_data = {}
        

        tuition_table = fee_content.find('table')
        if tuition_table:
            headers = [th.get_text(strip=True) for th in tuition_table.find_all('th')]
            tuition_data = []
            for row in tuition_table.find_all('tr')[1:]: 
                cols = row.find_all('td')
                if len(cols) == len(headers):
                    tuition_data.append({headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))})
            if tuition_data:
                fee_data["tuition_fees"] = tuition_data
        
       
        activities_text = fee_content.find(string=re.compile(r'Student Activities Fund.*Rs[\s\d,]+'))
        if activities_text:
            activities_match = re.search(r'Student Activities Fund.*?Rs[\s]*([\d,]+)', str(activities_text))
            if activities_match:
                fee_data["student_activities_fund"] = f"Rs {activities_match.group(1)} per semester"
        
    
        misc_fees = {}
        misc_section = fee_content.find(string=re.compile(r'Miscellaneous Fees'))
        if misc_section:
            misc_parent = misc_section.find_parent()
            if misc_parent:
              
                next_elem = misc_parent.find_next()
                while next_elem and not next_elem.find('table'):
                    if next_elem.name in ['p', 'div', 'li']:
                        text = next_elem.get_text(strip=True)
                        if text and 'Rs' in text and not any(admission_word in text.lower() for admission_word in ['admission', 'applicant']):
                         
                            fee_match = re.search(r'([^Rs]+)\s+Rs[\s\.]*([\d,]+)', text)
                            if fee_match:
                                fee_name = fee_match.group(1).strip()
                                fee_amount = f"Rs {fee_match.group(2)}"
                                if 'admission' not in fee_name.lower():
                                    misc_fees[fee_name] = fee_amount
                    next_elem = next_elem.find_next()
        
       
        misc_tables = fee_content.find_all('table')
        if len(misc_tables) > 1: 
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
        
      
        late_payment_text = fee_content.find(string=re.compile(r'Late Payment Fine'))
        if late_payment_text:
            late_parent = late_payment_text.find_parent()
            if late_parent:
                late_info = late_parent.find_next('p')
                if late_info:
                    fee_data["late_payment_fine"] = late_info.get_text(strip=True)
        
     
        refund_data = {}
        refund_section = fee_content.find(string=re.compile(r'Fee Refund Policy'))
        if refund_section:
            refund_parent = refund_section.find_parent()
            if refund_parent:
              
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
            fee_data["payment_methods"] = payment_methods[:5]
        
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

@web_scraping_bp.route('/feesfast', methods=['GET'])
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

@web_scraping_bp.route('/scholarshipsnust', methods=['GET'])
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


SISGP_URL = "https://www.ilmkidunya.com/scholarships/sisgp-scholarships"
@web_scraping_bp.route("/sisgp", methods=["GET"])
def scrape_sisgp():
    try:
        response = requests.get(SISGP_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

  
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

        
        intro_paragraphs = []
        intro_p = soup.find("p")
        if intro_p:
            intro_paragraphs = [p.get_text(strip=True) for p in intro_p.find_all_next("p", limit=3) if len(p.get_text(strip=True)) > 50]

  
        why_apply_points = []
        why_section = soup.find("h2", string=lambda t: t and "Why Pakistani Professionals Should Apply" in t)
        if why_section:
            ul = why_section.find_next("ul")
            if ul:
                why_apply_points = [li.get_text(strip=True) for li in ul.find_all("li")]

  
        eligibility_points = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Eligibility Criteria" in t)
        if eligibility_section:
            ul = eligibility_section.find_next("ul")
            if ul:
                eligibility_points = [li.get_text(strip=True) for li in ul.find_all("li")]


        coverage_points = []
        coverage_section = soup.find("h2", string=lambda t: t and "What Does the Scholarship Cover?" in t)
        if coverage_section:
            ul = coverage_section.find_next("ul")
            if ul:
                coverage_points = [li.get_text(strip=True) for li in ul.find_all("li")]

 
        apply_steps = []
        apply_section = soup.find("h2", string=lambda t: t and "How to Apply" in t)
        if apply_section:
            ol = apply_section.find_next("ol")
            if ol:
                apply_steps = [li.get_text(strip=True) for li in ol.find_all("li")]
            else:
      
                steps = apply_section.find_all_next("p", limit=10)
                apply_steps = [s.get_text(strip=True) for s in steps if len(s.get_text(strip=True)) > 30 and any(num in s.get_text() for num in ['1.', '2.', '3.', '4.'])]


        dates = []
        dates_section = soup.find("h2", string=lambda t: t and "Key Dates" in t)
        if dates_section:
            ul = dates_section.find_next("ul")
            if ul:
                dates = [li.get_text(strip=True) for li in ul.find_all("li")]


        benefits_points = []
        benefits_section = soup.find("h2", string=lambda t: t and "Benefits" in t)
        if benefits_section:
            ul = benefits_section.find_next("ul")
            if ul:
                benefits_points = [li.get_text(strip=True) for li in ul.find_all("li")]


        final_thoughts = []
        final_section = soup.find("h2", string=lambda t: t and "Final Thoughts" in t)
        if final_section:
            final_paragraphs = final_section.find_all_next("p", limit=3)
            final_thoughts = [p.get_text(strip=True) for p in final_paragraphs if len(p.get_text(strip=True)) > 30]

  
        sharing_buttons = []
        sharing_section = soup.find_all("button", string=lambda t: t and any(platform in t.lower() for platform in ["whatsapp", "facebook", "twitter", "linkedin", "pinterest", "email"]))
        if sharing_section:
            sharing_buttons = [btn.get_text(strip=True).lower() for btn in sharing_section]

      
        feedback_options = []
        feedback_section = soup.find("div", string=lambda t: t and "Is this page helpful?" in t)
        if feedback_section:
            feedback_buttons = feedback_section.find_all_next("button", limit=2)
            feedback_options = [btn.get_text(strip=True) for btn in feedback_buttons if btn.get_text(strip=True) in ["Yes", "No"]]

        
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


TURKIYE_URL = "https://www.ilmkidunya.com/scholarships/turkiye-burslari-scholarships"
@web_scraping_bp.route("/turkiye", methods=["GET"])
def scrape_turkiye():
    try:
        response = requests.get(TURKIYE_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

       
        intro_p = soup.find("p")
        intro_text = intro_p.get_text(strip=True) if intro_p else "No Intro Found"

        
        why_section = soup.find("h2", string=lambda t: t and "Why Türkiye Scholarships Are a Game-Changer" in t)
        why_points = []
        if why_section:
            ul = why_section.find_next("ul")
            if ul:
                why_points = [li.get_text(strip=True) for li in ul.find_all("li")]

        eligibility_academic = []
        eligibility_age = []
        eligibility_other = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Who is Eligible?" in t)
        if eligibility_section:
          
            academic_h3 = eligibility_section.find_next("h3", string=lambda t: t and "Academic Requirements" in t)
            if academic_h3:
                ul = academic_h3.find_next("ul")
                if ul:
                    eligibility_academic = [li.get_text(strip=True) for li in ul.find_all("li")]
          
            age_h3 = eligibility_section.find_next("h3", string=lambda t: t and "Age Limits" in t)
            if age_h3:
                ul = age_h3.find_next("ul")
                if ul:
                    eligibility_age = [li.get_text(strip=True) for li in ul.find_all("li")]
          
            other_h3 = eligibility_section.find_next("h3", string=lambda t: t and "Other Requirements" in t)
            if other_h3:
                ul = other_h3.find_next("ul")
                if ul:
                    eligibility_other = [li.get_text(strip=True) for li in ul.find_all("li")]

       
        benefits_table = soup.find("table")
        benefits = {}
        if benefits_table:
            rows = benefits_table.find_all("tr")[1:]  
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)
                    benefits[key] = value

       
        apply_steps = []
        apply_section = soup.find("h2", string=lambda t: t and "How to Apply" in t)
        if apply_section:
            ol = apply_section.find_next("ol")
            if ol:
                apply_steps = [li.get_text(strip=True) for li in ol.find_all("li")]
            else:
               
                steps = apply_section.find_all_next("p", limit=10)
                apply_steps = [s.get_text(strip=True) for s in steps if len(s.get_text(strip=True)) > 30 and any(num in s.get_text() for num in ['1.', '2.', '3.', '4.'])]

       
        selection_steps = []
        selection_section = soup.find("h2", string=lambda t: t and "Selection Process" in t)
        if selection_section:
            ul = selection_section.find_next("ul")
            if ul:
                selection_steps = [li.get_text(strip=True) for li in ul.find_all("li")]

       
        dates = []
        dates_section = soup.find("h2", string=lambda t: t and "Key Dates" in t)
        if dates_section:
            ul = dates_section.find_next("ul")
            if ul:
                dates = [li.get_text(strip=True) for li in ul.find_all("li")]

   
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


STIPENDIUM_URL = "https://www.ilmkidunya.com/scholarships/stipendium-hungaricum-scholarships"
@web_scraping_bp.route("/hungary", methods=["GET"])
def scrape_stipendium():
    try:
        response = requests.get(STIPENDIUM_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

       
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

       
        intro_paragraphs = []
        intro_section = soup.find("h2", string=lambda t: t and "Introduction" in t)
        if intro_section:
            intro_paragraphs = [p.get_text(strip=True) for p in intro_section.find_all_next("p", limit=3) if len(p.get_text(strip=True)) > 50]

      
        what_is = []
        what_section = soup.find("h2", string=lambda t: t and "What is the Stipendium Hungaricum Scholarship?" in t)
        if what_section:
            what_paragraphs = what_section.find_all_next("p", limit=3)
            what_is = [p.get_text(strip=True) for p in what_paragraphs if len(p.get_text(strip=True)) > 30]

        
        programs = []
        programs_section = soup.find("h2", string=lambda t: t and "Programs Offered" in t)
        if programs_section:
            ul = programs_section.find_next("ul")
            if ul:
                programs = [li.get_text(strip=True) for li in ul.find_all("li")]

      
        eligibility = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Eligibility Criteria" in t)
        if eligibility_section:
            ul = eligibility_section.find_next("ul")
            if ul:
                eligibility = [li.get_text(strip=True) for li in ul.find_all("li")]

  
        benefits = []
        benefits_section = soup.find("h2", string=lambda t: t and "Scholarship Benefits" in t)
        if benefits_section:
            ul = benefits_section.find_next("ul")
            if ul:
                benefits = [li.get_text(strip=True) for li in ul.find_all("li")]

        
        apply_steps = []
        apply_section = soup.find("h2", string=lambda t: t and "How to Apply" in t)
        if apply_section:
            steps = apply_section.find_all_next("p", limit=10)
            apply_steps = [s.get_text(strip=True) for s in steps if len(s.get_text(strip=True)) > 30]

       
        dates = []
        dates_section = soup.find("h2", string=lambda t: t and "Important Dates" in t)
        if dates_section:
            ul = dates_section.find_next("ul")
            if ul:
                dates = [li.get_text(strip=True) for li in ul.find_all("li")]

      
        why_apply = []
        why_section = soup.find("h2", string=lambda t: t and "Why Pakistani Students Should Apply" in t)
        if why_section:
            ul = why_section.find_next("ul")
            if ul:
                why_apply = [li.get_text(strip=True) for li in ul.find_all("li")]

       
        contact_info = []
        contact_section = soup.find("h2", string=lambda t: t and "Contact for Queries" in t)
        if contact_section:
            contact_paragraphs = contact_section.find_all_next("p", limit=3)
            contact_info = [p.get_text(strip=True) for p in contact_paragraphs if len(p.get_text(strip=True)) > 20]

        
        conclusion = []
        conclusion_section = soup.find("h2", string=lambda t: t and "Conclusion" in t)
        if conclusion_section:
            conclusion_paragraphs = conclusion_section.find_all_next("p", limit=3)
            conclusion = [p.get_text(strip=True) for p in conclusion_paragraphs if len(p.get_text(strip=True)) > 30]

        
        sharing_buttons = []
        sharing_section = soup.find_all("button", string=lambda t: t and any(platform in t.lower() for platform in ["whatsapp", "facebook", "twitter", "linkedin", "pinterest", "email"]))
        if sharing_section:
            sharing_buttons = [btn.get_text(strip=True).lower() for btn in sharing_section]

   
        feedback_options = []
        feedback_section = soup.find("div", string=lambda t: t and "Is this page helpful?" in t)
        if feedback_section:
            feedback_buttons = feedback_section.find_all_next("button", limit=2)
            feedback_options = [btn.get_text(strip=True) for btn in feedback_buttons if btn.get_text(strip=True) in ["Yes", "No"]]

 
        community_info = []
        community_sections = soup.find_all("p", string=lambda t: t and any(keyword in t.lower() for keyword in ["followers", "subscribers", "join", "follow us"]))
        for section in community_sections:
            text = section.get_text(strip=True)
            if len(text) > 20:
                community_info.append(text)

       
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

CHEVENING_URL = "https://www.ilmkidunya.com/scholarships/chevening-scholarships"
@web_scraping_bp.route("/chevening", methods=["GET"])
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


ERASMUS_URL = "https://www.ilmkidunya.com/scholarships/erasmus-mundus-scholarships"
@web_scraping_bp.route("/erasmus", methods=["GET"])
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


COMMONWEALTH_URL = "https://www.ilmkidunya.com/scholarships/commonwealth-international-scholarships"
@web_scraping_bp.route("/commonwealth", methods=["GET"])
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


RHODES_URL = "https://www.ilmkidunya.com/scholarships/rhodes-uk-scholarships"
@web_scraping_bp.route("/rhodes", methods=["GET"])
def scrape_rhodes():
    try:
        response = requests.get(RHODES_URL, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

     
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No Title Found"

       
        introduction = ""
        intro_section = soup.find("h2", string=lambda t: t and "Introduction" in t)
        if intro_section:
            next_p = intro_section.find_next("p")
            if next_p:
                introduction = next_p.get_text(strip=True)

        
        facts = {}
        quick_facts = soup.find_all("tr")
        for row in quick_facts:
            cols = row.find_all(["td", "th"])
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                facts[key] = value

     
        benefits = []
        benefits_section = soup.find("h2", string=lambda t: t and "What the Scholarship Covers" in t)
        if benefits_section:
            ul = benefits_section.find_next("ul")
            if ul:
                benefits = [li.get_text(strip=True) for li in ul.find_all("li")]
            else:
               
                next_p = benefits_section.find_next("p")
                if next_p:
                    benefits = [next_p.get_text(strip=True)]

       
        eligibility = []
        eligibility_section = soup.find("h2", string=lambda t: t and "Eligibility Criteria" in t)
        if eligibility_section:
            ul = eligibility_section.find_next("ul")
            if ul:
                eligibility = [li.get_text(strip=True) for li in ul.find_all("li")]

   
        application_process = []
        application_section = soup.find("h2", string=lambda t: t and "Application Process" in t)
        if application_section:
          
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

     
        why_apply = []
        why_apply_section = soup.find("h2", string=lambda t: t and "Why Apply for Rhodes" in t)
        if why_apply_section:
            ul = why_apply_section.find_next("ul")
            if ul:
                why_apply = [li.get_text(strip=True) for li in ul.find_all("li")]

       
        tips = []
        tips_section = soup.find("h2", string=lambda t: t and "Tips for a Strong Application" in t)
        if tips_section:
            ul = tips_section.find_next("ul")
            if ul:
                tips = [li.get_text(strip=True) for li in ul.find_all("li")]

    
        final_thoughts = ""
        final_section = soup.find("h2", string=lambda t: t and "Final Words" in t) or soup.find("h2", string=lambda t: t and "Final Thoughts" in t)
        if final_section:
            next_p = final_section.find_next("p")
            if next_p:
                final_thoughts = next_p.get_text(strip=True)

    
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