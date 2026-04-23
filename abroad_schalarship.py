from flask import Flask, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)  # Enable CORS so it can run on any device / frontend

# -----------------------
# SI Scholarship for Global Professionals Scraper
# -----------------------
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
# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)