
from flask import Blueprint, jsonify
import requests
from bs4 import BeautifulSoup
import re
import time


events_bp = Blueprint('events', __name__)

def fetch_events(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        events = []
        for a in soup.find_all('a'):
            text = a.get_text(strip=True)
            if 'Event Date:' in text:
                parts = text.split('Event Date:')
                title = parts[0].strip()
                date = parts[1].strip() if len(parts) > 1 else None
                events.append({'title': title, 'date': date})
        return events
    except Exception as e:
        print(f"Error fetching events from {url}: {e}")
        return [{'title': f'Error: {str(e)}', 'date': None}]

def fetch_neduet_events(page=3):
    try:
        url = f'https://www.neduet.edu.pk/content/events?page={page}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        content = soup.find('div', {'class': 'content'})
        if not content:
            content = soup
        text_lines = [t.strip() for t in content.stripped_strings]
        events = []
        date_pattern = re.compile(
            r'(\b\d{1,2}(?:st|nd|rd|th)?\s+\w+,\s*\d{4}\b|\b\w+\s+\d{1,2}\s*-\s*\d{1,2},?\s*\d{4}\b|\b\d{1,2}\s*-\s*\d{1,2}\s*\w+\s*\d{4}\b)'
        )
        for i, line in enumerate(text_lines):
            if date_pattern.search(line):
                if i + 1 < len(text_lines):
                    title = text_lines[i + 1]
                    date = line
                    if not any(word in title.lower() for word in ["breadcrumb", "home", "events", "pagination", "quick", "links"]):
                        events.append({'date': date, 'title': title})
        return events
    except Exception as e:
        print(f"Error fetching NEDUET events: {e}")
        return [{'title': f'Error: {str(e)}', 'date': None}]

def fetch_uet_taxila_events():
    try:
        url = 'https://www.uettaxila.edu.pk/Events/All'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        events = []

        table = soup.find('table')
        if not table:
            return [{'title': 'No events table found', 'date': None}]

        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 2:
                title = cols[0].get_text(strip=True)
                date = cols[1].get_text(strip=True)
                events.append({'title': title, 'date': date})
        return events if events else [{'title': 'No events found', 'date': None}]
    except Exception as e:
        print(f"Error fetching UET Taxila events: {e}")
        return [{'title': f'Error: {str(e)}', 'date': None}]

# Event routes with better error handling
@events_bp.route('/api/comsats_events', methods=['GET'])
def comsats_events():
    try:
        url = 'https://ww2.comsats.edu.pk/alumni/allevents.aspx'
        events = fetch_events(url)
        return jsonify({
            'success': True,
            'events': events,
            'count': len(events)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'events': []
        }), 500

@events_bp.route('/api/neduet_events', methods=['GET'])
def neduet_events():
    try:
        events = fetch_neduet_events(3)
        return jsonify({
            'success': True,
            'events': events,
            'count': len(events)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'events': []
        }), 500

@events_bp.route('/api/uet_taxila_events', methods=['GET'])
def uet_taxila_events():
    try:
        events = fetch_uet_taxila_events()
        return jsonify({
            'success': True,
            'events': events,
            'count': len(events)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'events': []
        }), 500