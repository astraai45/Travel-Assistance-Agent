It seems that the markdown links may not be working as expected due to formatting issues. I'll provide a version where the contributor names are properly linked using markdown syntax.

Here’s the corrected version of the README:

````markdown
# Travel Assistant Agent 🌍✈️

This project provides an **AI-powered Travel Assistant** that helps users plan personalized trips based on their preferences. It leverages **Ollama** for language understanding, **Chroma** for vector search, and **Google Translate** for language translation. Users can input their preferences via text or voice (in multiple languages), upload PDF documents containing travel-related information, and get a fully personalized travel itinerary, including flights, accommodations, activities, and dining options.

---

## 🛠️ Features

- **Voice Input**: Allows users to interact with the assistant via voice input.
- **Multi-Language Support**: The assistant can respond in **English**, **Telugu**, and **Tamil** based on user preferences.
- **PDF Travel Preferences**: Users can upload PDF documents (e.g., travel brochures, itineraries) that are ingested into the system for personalized recommendations.
- **Chat Interface**: A real-time chat interface where users can ask travel-related questions.
- **Personalized Travel Itinerary**: The assistant generates a complete itinerary based on the user's preferences, including:
  - Flights
  - Accommodation
  - Activities (sightseeing, adventure)
  - Dining options (restaurants, cafes)

---

## 🚀 Technologies Used

- **Streamlit**: For creating an interactive web application.
- **Langchain**: For working with language models and document embeddings.
- **Ollama**: For processing natural language input and generating responses.
- **Chroma**: For vector store management and document retrieval.
- **Google Translate**: For multilingual support and translation of responses.
- **PyPDF2**: For reading and extracting text from PDF documents.
- **SpeechRecognition**: For converting voice input into text.
- **Deep Translator**: For language translation.

---

## 🧑‍💻 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/travel-assistant-agent.git
   cd travel-assistant-agent
````

2. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. Open your browser and go to the local Streamlit app, usually hosted at `http://localhost:8501/`.

---

## 💡 Usage

### 1. **Upload Travel Preferences (PDF)**

* Upload your travel-related PDF (it could be an itinerary, travel brochure, etc.) on the sidebar.
* The assistant will ingest the content of the PDF, process it, and store the relevant information in the vector store for future reference.

### 2. **Chat with the Assistant**

* **Text Input**: You can type your query in the text box.
* **Voice Input**: Check the "Enable Voice Input" checkbox on the sidebar to record and ask questions using your voice.
* You can ask about:

  * Travel destinations
  * Hotels and resorts
  * Activities (adventure, sightseeing)
  * Dining options
  * Flights and other travel-related questions.

### 3. **Language Preferences**

* You can select your preferred language from the sidebar (English, Telugu, Tamil).
* Responses will be automatically translated into the chosen language.

---

## 🛠️ Customization

* Modify the prompt in `app.py` to adjust the assistant's tone or behavior.
* You can update the `language model` in the code to use different models from Ollama.
* If you have additional data sources or want to use different document formats, you can modify the document ingestion pipeline.

---

## 🤖 Example Interactions

* **User**: "Can you help me plan a trip to Goa?"

  * **Assistant**: "Sure! What’s your budget and interests for the trip?"
* **User**: "What are the best hotels in Goa?"

  * **Assistant**: "Based on the preferences you uploaded, here are some top hotels in Goa: \[List of hotels]."

---



---

## 👥 Contributors

* [Pavan Balaji](https://github.com/pavanbalaji45)
* [Balaji Kartheek](https://github.com/Balaji-Kartheek)

---

## 📝 Acknowledgements

* Thanks to **Langchain**, **Ollama**, and **Chroma** for providing the essential tools for document retrieval and language processing.
* **Streamlit** for enabling easy and interactive web app development.

```

This should now work properly when pasted into your `README.md` file. The contributor links to **Pavan Balaji** and **Balaji Kartheek** are correctly formatted with markdown syntax.
```
