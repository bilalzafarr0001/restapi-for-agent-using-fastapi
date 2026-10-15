import os

# pytest runs this file first. The agents need API keys to start, and the tests
# never call the real APIs, so fake keys are enough.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("SERPER_API_KEY", "test-key")
os.environ.setdefault("WEATHER_API_KEY", "test-key")
