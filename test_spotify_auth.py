# test_spotify_auth.py
from spotify_utils import search_tracks_by_mood

try:
    print("Testing Spotify API search for 'happy'...")
    tracks = search_tracks_by_mood("happy", limit=2)
    
    if tracks:
        print("✅ SUCCESS: Spotify API is connected and returned tracks.")
        print(f"Track 1: {tracks[0]['name']} by {tracks[0]['artist']}")
    else:
        print("❌ FAILURE: Spotify API connected but returned no tracks.")

except Exception as e:
    # This will print the raw API error (e.g., 401 Unauthorized)
    print(f"❌ AUTHENTICATION FAILED. The specific error is: {e}")
