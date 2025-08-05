from llama_parse import LlamaParse
import json
from llama_cloud_services import LlamaExtract
from dotenv import load_dotenv
import os
load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
# Your schema as a Python dict (copy-paste the JSON but convert keys/strings)
schema = {
  "additionalProperties": False,
  "properties": {
    "page_content": {
      "description": "List of content blocks for a page.",
      "items": {
        "additionalProperties": False,
        "properties": {
          "block_type": {
            "enum": [
              "title",
              "paragraph",
              "list",
              "table",
              "columnar_text",
              "footnote"
            ],
            "type": "string",
            "description": "Type of content block."
          },
          "content": {
            "description": "Text content for title, paragraph, or footnote blocks. Empty string if not applicable.",
            "type": "string"
          },
          "items": {
            "description": "List items for list-type blocks. Empty array if not applicable.",
            "items": {
              "type": "string"
            },
            "type": "array"
          },
          "table_title": {
            "description": "Optional title above the table. Empty string if not present.",
            "type": "string"
          },
          "headers": {
            "description": "Column headers for table blocks. Empty array if not applicable.",
            "items": {
              "type": "string"
            },
            "type": "array"
          },
          "rows": {
            "description": "Each table row as an array of cell texts. Empty array if no rows.",
            "items": {
              "items": {
                "type": "string"
              },
              "type": "array"
            },
            "type": "array"
          },
          "columns": {
            "description": "Used for multi-column text (like Arabic RTL). Empty array if not applicable.",
            "items": {
              "additionalProperties": False,
              "properties": {
                "column_id": {
                  "type": "integer"
                },
                "content": {
                  "type": "string"
                }
              },
              "required": [
                "column_id",
                "content"
              ],
              "type": "object"
            },
            "type": "array"
          }
        },
        "required": [
          "block_type",
          "content",
          "items",
          "table_title",
          "headers",
          "rows",
          "columns"
        ],
        "type": "object"
      },
      "type": "array"
    }
  },
  "required": [
    "page_content"
  ],
  "type": "object"
}
extractor = LlamaExtract(api_key=LLAMA_CLOUD_API_KEY)
agent = extractor.create_agent(name="arab_pages", data_schema=schema,config={"extraction_mode": "MULTIMODAL",
        "extraction_target": "PER_PAGE", "cite_sources": False,
        "use_reasoning": False,          
        "confidence_scores": True})        
result = agent.extract("Documents/ksa-personal-data-protection-law-series-part-1-ar.pdf")
# Convert the result to JSON string for writing to file
with open("Test_Arabic/arabic_llama.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(result.data, ensure_ascii=False, indent=2))
        f.write("\n" + "="*50 + "\n")  # Add separator for multiple runs
print("Extraction completed successfully!")