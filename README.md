
Advanced RAG with CrewAI
This project implements an advanced Retrieval-Augmented Generation (RAG) system with voice output capabilities using CrewAI. It uses a crew-based approach to process queries, classify intents, and provide appropriate responses.
Repository
https://github.com/asheesh4545/advance-rag-crewai
Features
* Intent classification for user queries
* Document summarization
* Context-based question answering
* Web search integration
* Text-to-speech output for responses
Components
The application consists of several key components:
1. Main application (app.py)
2. Helper functions (helper.py)
3. Sound utilities (soundutils.py)
Prerequisites
* Python 3.7+
* Groq API key
* Tavily API key
* Sarvam AI API key
Installation
1. Clone the repository:


Copy
Copy
git clone https://github.com/asheesh4545/advance-rag-crewai.git cd advance-rag-crewai
2. Install the required packages:


Copy
Copy
pip install -r requirements.txt
3. Set up environment variables: Create a .env file in the project root and add your API keys:


Copy
Copy
GROQ_API_KEY=your_groq_api_key TAVILY_API_KEY=your_tavily_api_key SARVAM_API_KEY=your_sarvam_api_key
Usage
Run the Streamlit app:


Copy
Copy
streamlit run app.py
Navigate to the provided URL in your web browser to interact with the application.
How It Works
1. Intent Classification: The system first classifies the user's query intent (greeting, document inquiry, or general question).
2. Query Processing: Based on the intent, one of three crews is activated:
   * Greeting Crew: Responds to greetings
   * Document Summary Crew: Provides comprehensive document summaries
   * QA Crew: Handles general questions using context retrieval, web search, and answer formulation
3. Voice Output: Responses are converted to speech and played back to the user.
Project Structure
* app.py: Main application file containing the Streamlit UI and core logic
* helper.py: Contains functions for document ingestion and retrieval
* soundutils.py: Handles text-to-speech conversion and audio playback
Agents
The system uses several specialized agents:
* Intent Classifier
* Greeter
* Document Summarizer
* Context Retriever
* Context Analyzer
* Web Searcher
* Answer Formulator
Sound Utilities (soundutils.py)
The soundutils.py file is responsible for handling the text-to-speech functionality of the application. Here's an overview of its main components:
1. Initialization: The initialize_audio function sets up the audio playback system, ensuring it's ready to output speech.
2. Text-to-Speech Conversion: The file interacts with the Sarvam AI API to convert text responses into speech.
3. Audio Processing: Once the speech audio is received from the API, it may be processed or formatted to ensure compatibility with the playback system.
4. Playback: The process_and_play_text function takes the generated speech audio and plays it back to the user. This function is called asynchronously for each task output in the main application.
5. Error Handling: The system includes error handling to manage potential issues with the API connection or audio playback, ensuring a smooth user experience.
To use the sound utilities effectively, ensure that your Sarvam AI API key is correctly set in the .env file and that your system's audio output is properly configured.
