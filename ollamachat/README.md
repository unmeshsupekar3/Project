# Ollamachat: README

## Overview
Ollamachat is a Python-based project designed to showcase the capabilities of Large Language Models (LLMs) and semantic similarity search through two distinct tasks:

1. **Task 1: Patient Data Extraction**
   - Extract structured information (name, gender, age, weight, height, BMI, and chief medical complaint) from unstructured patient text using an LLM.

2. **Task 2: Semantic Similarity Search**
   - Fetch, process, and chunk a Wikipedia article, store its content in a semantic similarity database, and allow queries for retrieving relevant results.

## Folder Structure

```
- patient
    - models: Contains Python scripts for core logic.
        - chat_ollama.py: Handles patient data extraction (Task 1).
        - wikipedia.py: Handles semantic similarity search (Task 2).
    - routes: FastAPI routes for backend integration.
    - schemas: Pydantic models for request/response validation.
    - streamlit_demo: Streamlit scripts for Task 1 and Task 2 demos.
    - app.py: Main entry point to run the backend API.
    - requirements.txt: Dependencies for the project.
```

## Task 1: Patient Data Extraction

### Description
Task 1 uses an LLM to process free-text patient data and extract structured information into a dictionary containing:
- Name
- Gender
- Age
- Weight
- Height
- BMI
- Chief Medical Complaint

### Example

#### Input:
```
John R. Whitaker, a 52-year-old male, stands 5'10" tall and weighs 198 lbs. He reports chronic lower back pain and shortness of breath.
```

#### Output:
```json
{
  "name": "John R. Whitaker",
  "gender": "male",
  "age": 52,
  "weight": 90,
  "height": 178,
  "BMI": null,
  "chief_medical_complaint": "chronic lower back pain and shortness of breath"
}
```

### Running Task 1

Run chat_ollama.py file from models folder or try alternatives below:

1. **API**:
   - Run the `app.py` file to start the backend server:
     ```bash
     python app.py
     ```
   - Use Postman or Swagger to send POST requests to the endpoint `/get_patient_details`

2. **Streamlit Demo**:
   - Navigate to the project directory and run the demo script:
     ```bash
     streamlit run patient/streamlit_demo/chat_ollama_stream.py
     ```

---

## Task 2: Semantic Similarity Search

### Description
Task 2 processes a Wikipedia article (e.g., about Munich) by:
1. Fetching the article using the Wikipedia API.
2. Chunking the content into manageable parts.
3. Storing the chunks in a semantic similarity database.
4. Allowing queries to retrieve the most relevant chunks.

### Example

#### Query:
```
Do people use bicycles in Munich?
```

#### Output:
```json
{
    "status_code": 200,
    "status": "success",
    "result": [
        "Cycling\nCycling has a strong presence in the city and is recognized as a good alternative. The growing number of bicycle lanes are widely used throughout the year. Cycle paths can be found alongside the majority of sidewalks and streets, although the newer or renovated ones are much easier to tell apart from pavements than older ones. A modern bike hire system is available within the area bounded by the Mittlerer Ring.\n\nCultural history trails and bicycle routes\nSince 2001, historically interesting places in Munich can be explored via the List of cultural history trails in Munich (KulturGeschichtsPfade). Sign-posted cycle routes are the Outer Äußere Radlring (outer cycle route) and the RadlRing München.",
        "Transport\nMunich has an extensive public transport system consisting of an underground metro, trams, buses and high-speed rail. In 2015, the transport modal share in Munich was 38 percent public transport, 25 percent car, 23 percent walking, and 15 percent bicycle. Its public transport system delivered 566 million passenger trips that year.\nMunich is the hub of a developed regional transportation system, including the second-largest airport in Germany and the Berlin–Munich high-speed railway, which connects Munich to the German capital city with a journey time of about 4 hours. Flixmobility which offers intercity coach service is headquartered in Munich.\nThe trade fair Transport Logistic is held every two years at the Neue Messe München (Messe München International).",
        "Public transport\nFor its urban population of 2.6 million people, Munich and its closest suburbs have a comprehensive network of public transport incorporating the Munich U-Bahn, the Munich S-Bahn, trams and buses. The system is supervised by the Munich Transport and Tariff Association (Münchner Verkehrs- und Tarifverbund). The Munich tramway is the oldest existing public transportation system in the city, which has been in operation since 1876. Munich also has an extensive network of bus lines. The average amount of time people spend commuting to and from work with public transit in Munich on a weekday is 56 min."
    ],
    "message": "Successfully extracted answer for the question"
}
```

### Running Task 2

Run wikipedia.py from models folder or try alternatives below:

1. **API**:
   - Run the `app.py` file to start the backend server:
     ```bash
     python app.py
     ```
   - Use Postman or Swagger to send POST requests to the endpoint `/get_munich`

2. **Streamlit Demo**:
   - Navigate to the project directory and run the demo script:
     ```bash
     streamlit run patient/streamlit_demo/wikipedia_stream.py
     ```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ollamachat
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## API Usage

### Starting the API
Run the following command to start the FastAPI server:
```bash
python app.py
```
The API will be accessible at `http://0.0.0.0:953`. Use Swagger UI (`/docs`) or Postman for testing.

### Endpoints

#### Task 1: Patient Data Extraction
- **Endpoint**: `/extract_patient_data`
- **Method**: `POST`
- **Payload**: Free-text string containing patient details.
- **Response**: JSON dictionary with extracted fields.

#### Task 2: Semantic Similarity Search
- **Endpoint**: `/semantic_search`
- **Method**: `POST`
- **Payload**: Query string.
- **Response**: JSON array of relevant chunks.

---

## Streamlit Applications

### Task 1 Demo
Run the following command:
```bash
streamlit run patient/streamlit_demo/chat_ollama_stream.py
```

### Task 2 Demo
Run the following command:
```bash
streamlit run patient/streamlit_demo/wikipedia_stream.py
```

---

## Key Dependencies
- **FastAPI**: Backend API framework.
- **Streamlit**: Interactive front-end application.
- **Wikipedia-API**: Fetch and parse Wikipedia articles.
- **nomic-embed-text**: Embedding for semantic similarity.
- **llama3.2**: Lightweight LLM for local execution.

---

## Conclusion
Ollamachat highlights the practical applications of LLMs in healthcare data extraction and semantic similarity search. Its modular design allows easy integration and extension. Streamlit demos offer an interactive experience for exploring project functionality.
