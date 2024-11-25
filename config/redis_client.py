import redis
import os

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    decode_responses=True
)

# Test connection
try:
    redis_client.ping()
    print("Connected to Redis!")
except redis.ConnectionError as e:
    print(f"Failed to connect to Redis: {e}")
