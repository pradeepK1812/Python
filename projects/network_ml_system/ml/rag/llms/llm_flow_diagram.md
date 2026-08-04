LLM flow diagram:

generate(prompt)
      │
      ▼
Build URL
      │
      ▼
Build JSON payload
      │
      ▼
requests.post(...)
      │
      ▼
response.raise_for_status()
      │
      ▼
response.json()
      │
      ▼
Extract "response"
      │
      ▼
Return generated text

--------------------------------------------------------------------------------------------------------------------------

Error Handling:


----------------------------------------------------------------------------------------------------------------
Connection
    │
    ▼
ConnectionError
    │
    ▼
RuntimeError

----------------------------------------

Timeout
    │
    ▼
Timeout
    │
    ▼
RuntimeError

----------------------------------------

HTTP
    │
    ▼
response.raise_for_status()
    │
    ▼
HTTPError
    │
    ▼
RuntimeError

----------------------------------------

JSON Parsing
    │
    ▼
ValueError
    │
    ▼
RuntimeError

----------------------------------------

Response Extraction
    │
    ▼
KeyError
    │
    ▼
RuntimeError

--------------------------------------------------------------------------------------------------------------------------------
