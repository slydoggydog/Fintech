"""
Gaming Schedule Coordinator
A simple web app to coordinate gaming nights with friends.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
import uuid
from datetime import datetime, timedelta

app = Flask(__name__)

# Data storage file
DATA_FILE = 'sessions.json'


def load_sessions():
    """Load sessions from JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_sessions(sessions):
    """Save sessions to JSON file."""
    with open(DATA_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)


def get_week_dates(start_date=None):
    """Get dates for the current or specified week (Monday to Sunday)."""
    if start_date is None:
        today = datetime.now()
        # Start from next Monday if it's already past Wednesday
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0 and today.weekday() == 0:
            start = today
        else:
            start = today + timedelta(days=days_until_monday)
    else:
        start = datetime.strptime(start_date, '%Y-%m-%d')

    dates = []
    for i in range(7):
        date = start + timedelta(days=i)
        dates.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_name': date.strftime('%A'),
            'display': date.strftime('%b %d')
        })
    return dates


@app.route('/')
def home():
    """Home page - create a new session."""
    return render_template('index.html')


@app.route('/create', methods=['POST'])
def create_session():
    """Create a new gaming session."""
    sessions = load_sessions()

    session_id = str(uuid.uuid4())[:8]
    creator_name = request.form.get('creator_name', 'Anonymous')
    session_name = request.form.get('session_name', 'Gaming Night')
    start_date = request.form.get('start_date')

    # Get week dates
    week_dates = get_week_dates(start_date)

    sessions[session_id] = {
        'id': session_id,
        'name': session_name,
        'creator': creator_name,
        'created_at': datetime.now().isoformat(),
        'week_dates': week_dates,
        'participants': {},
    }

    save_sessions(sessions)

    return redirect(url_for('vote', session_id=session_id, name=creator_name))


@app.route('/session/<session_id>')
def view_session(session_id):
    """View a session and vote."""
    sessions = load_sessions()

    if session_id not in sessions:
        return render_template('error.html', message='Session not found!'), 404

    session = sessions[session_id]
    return render_template('session.html', session=session)


@app.route('/session/<session_id>/vote', methods=['GET', 'POST'])
def vote(session_id):
    """Vote on available nights."""
    sessions = load_sessions()

    if session_id not in sessions:
        return render_template('error.html', message='Session not found!'), 404

    session = sessions[session_id]

    if request.method == 'POST':
        name = request.form.get('name', 'Anonymous')
        selected_dates = request.form.getlist('dates')

        session['participants'][name] = {
            'name': name,
            'available_dates': selected_dates,
            'voted_at': datetime.now().isoformat()
        }

        save_sessions(sessions)

        return redirect(url_for('results', session_id=session_id))

    name = request.args.get('name', '')
    return render_template('vote.html', session=session, name=name)


@app.route('/session/<session_id>/results')
def results(session_id):
    """View results - which nights work best."""
    sessions = load_sessions()

    if session_id not in sessions:
        return render_template('error.html', message='Session not found!'), 404

    session = sessions[session_id]

    # Calculate availability counts
    date_counts = {}
    for date_info in session['week_dates']:
        date = date_info['date']
        date_counts[date] = {
            'date': date,
            'day_name': date_info['day_name'],
            'display': date_info['display'],
            'count': 0,
            'voters': []
        }

    for participant_name, participant_data in session['participants'].items():
        for date in participant_data['available_dates']:
            if date in date_counts:
                date_counts[date]['count'] += 1
                date_counts[date]['voters'].append(participant_name)

    # Find the max count to highlight best nights
    max_count = max((d['count'] for d in date_counts.values()), default=0)
    total_participants = len(session['participants'])

    # Find nights where everyone can make it
    perfect_nights = [d for d in date_counts.values() if d['count'] == total_participants and total_participants > 0]

    return render_template('results.html',
                         session=session,
                         date_counts=date_counts,
                         max_count=max_count,
                         total_participants=total_participants,
                         perfect_nights=perfect_nights)


@app.route('/api/sessions/<session_id>')
def api_session(session_id):
    """API endpoint to get session data."""
    sessions = load_sessions()

    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify(sessions[session_id])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
