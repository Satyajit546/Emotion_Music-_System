import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

def get_spotify_client():
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        raise Exception("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    return Spotify(auth_manager=auth_manager)

def search_tracks_by_mood(mood, limit=10):
    sp = get_spotify_client()
    q = mood + " mood"
    res = sp.search(q=q, type='track', limit=limit)
    tracks = []
    for item in res['tracks']['items']:
        tracks.append({
            "name": item['name'],
            "artist": ", ".join([a['name'] for a in item['artists']]),
            "preview_url": item['preview_url'],
            "spotify_url": item['external_urls']['spotify']
        })
    return tracks
