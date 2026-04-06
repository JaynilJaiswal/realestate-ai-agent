import requests
import time
import random
import uuid

# Your live Cloud Run endpoint
URL = "https://realestate-backend-538802495910.us-west2.run.app/api/v1/query"

# A mix of queries designed to trigger different tools and execution paths
PROMPTS = [
    # 1. Triggers Qdrant Database Search
    "Find me a highly-rated 2 bedroom apartment under $150 a night.",
    "Looking for a cheap place to stay in the downtown area.",
    "Show me the most expensive luxury rental available.",
    
    # 2. Triggers Live Web Search & Scraping
    "What is the current average 30-year fixed mortgage rate today?",
    "Search the web for news about the housing market in North Carolina.",
    
    # 3. Triggers Math Evaluator
    "Calculate the monthly mortgage payment for a $350,000 house with 20% down, 6.5% interest, over 30 years.",
    "If I buy a property for $500k and put 10% down, how much is the down payment?",
    
    # 4. Complex Multi-Tool Chaining
    "Find a 3-bedroom listing under $200 a night. Then, assuming I bought it for $300,000 with 20% down at a 7% interest rate for 30 years, calculate my monthly mortgage.",
    "Search the web for current interest rates, then calculate a 30-year mortgage on a $400k home using that rate with 20% down.",
    
    # 5. Edge Cases (To test error handling/LLM guardrails)
    "Calculate the mortgage for a house that costs potato dollars.",
    "Just say hello and tell me what you can do."
]

def generate_traffic(num_requests=15):
    print(f"🚀 Starting Synthetic Traffic Generator for {URL}")
    print(f"🎯 Target: {num_requests} requests. Pacing to respect Groq limits...\n")
    
    # Generate a few fake users to simulate different sessions
    fake_users = [f"user_{str(uuid.uuid4())[:8]}" for _ in range(3)]
    
    for i in range(num_requests):
        # Pick a random prompt and a random user
        prompt = random.choice(PROMPTS)
        session_id = random.choice(fake_users)
        
        payload = {
            "session_id": session_id,
            "user_input": prompt
        }
        
        print(f"[{i+1}/{num_requests}] 👤 {session_id} asking:")
        print(f"   💬 '{prompt}'")
        
        start_time = time.time()
        
        try:
            # Send request with a 60-second timeout (Cloud Run cold starts + LLM thinking)
            response = requests.post(URL, json=payload, timeout=60)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   ✅ Success ({elapsed:.2f}s)")
            else:
                print(f"   🚨 Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   💥 Connection Failed: {e}")
            
        # THE GOLDEN RULE OF FREE TIERS: Add Jitter & Sleep
        # Sleep between 10 to 15 seconds to ensure we stay under ~30 RPM Groq limit
        sleep_time = random.uniform(10.0, 15.0)
        print(f"   ⏳ Sleeping for {sleep_time:.1f}s to avoid rate limits...\n")
        time.sleep(sleep_time)

    print("🏁 Traffic generation complete! Go check your Grafana dashboard.")

if __name__ == "__main__":
    # 15 requests * ~12 seconds each = ~3 minutes to run. 
    # This generates enough data to populate charts without hitting any free tier ceilings.
    generate_traffic(num_requests=15)