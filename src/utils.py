
from pymongo import MongoClient
import streamlit as st

MONGO_URI = st.secrets['mongo_uri']
# --- Cosmos DB Connection ---
client = MongoClient(MONGO_URI)
db = client["llmops_db"]
logs_collection = db["query_logs"]

    
def store_logs_ToCosmosDB(qa_log_entry):
  
    # --- Store Log in Cosmos DB ---
    logs_collection.insert_one(qa_log_entry)


def update_feedback(session_id: str, feedback_value: str):
    """
    Update the 'feedback' field for a specific session_id in Cosmos DB.
    
    feedback_value: should be 'like' or 'dislike'
    """
    result = logs_collection.update_one(
        {"session_id": session_id},
        {"$set": {"feedback": feedback_value}}
    )
    return result.modified_count > 0  # True if updated

