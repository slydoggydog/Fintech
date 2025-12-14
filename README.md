# Game Night Coordinator

A simple web app to coordinate gaming nights with your friends. Create a poll, share the link, and find out which nights work best for everyone!

## Features

- Create gaming session polls with custom names
- Share links with friends
- Everyone picks their available nights
- See results with highlighted best nights
- Perfect matches (when everyone's free) are celebrated!

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python app.py
   ```

3. Open http://localhost:5000 in your browser

## How to Use

1. **Create a Poll**: Enter your name and a session name (e.g., "Weekly Valorant Night")
2. **Share the Link**: Copy the link and send it to your friends
3. **Vote**: Everyone selects the nights they're available
4. **View Results**: See which nights work best - perfect matches are highlighted!

## Deploying

To share with friends over the internet, you can:

- Deploy to a free service like [Render](https://render.com), [Railway](https://railway.app), or [PythonAnywhere](https://www.pythonanywhere.com)
- Use [ngrok](https://ngrok.com) to temporarily expose your local server
- Host on any VPS or cloud provider

## Tech Stack

- Python Flask (backend)
- HTML/CSS/JavaScript (frontend)
- JSON file storage (simple, no database needed)
