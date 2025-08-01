# LangChain Q&A Chatbot with Conversation History

A sophisticated Q&A chatbot application built with LangChain, Streamlit, and Google's Generative AI (Gemini) that maintains conversation history across multiple chat sessions.

## 🌟 Features

- **Multi-Session Chat Management**: Create and switch between multiple conversation sessions
- **Persistent Conversation History**: Each session maintains its own conversation history
- **Intelligent Session Summaries**: Automatic generation of chat summaries for easy session identification
- **Real-time Chat Interface**: Clean, intuitive Streamlit-based user interface
- **Google Gemini Integration**: Powered by Google's Gemini 1.5 Flash model for intelligent responses
- **Session Persistence**: Chat sessions persist during the application runtime

## 📁 Project Structure

```
LangChain/
├── Q&A_Chatbot/
│   ├── QA_Chatbot.py              # Main Streamlit application
│   └── history/
│       ├── __init__.py
│       ├── Message_History.py     # Chat history management class
│       └── __pycache__/
├── Basics/                        # Basic LangChain examples and tutorials
├── RAG_Document_Bot/              # Document-based RAG chatbot
├── Research_Papers/               # PDF documents for RAG
├── requirements.txt               # Project dependencies
├── Chat_History.ipynb             # Jupyter notebook for experimentation
├── FastAPI_LangServe_LangChain.py # FastAPI integration example
└── Streamlit_Translater.py        # Translation application
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API key for Gemini AI
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd LangChain
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

1. **Start the Q&A Chatbot:**
   ```bash
   cd Q&A_Chatbot
   streamlit run QA_Chatbot.py
   ```

2. **Open your browser:**
   The application will open automatically at `http://localhost:8501`

## 💡 How to Use

### Starting a New Chat
1. Click the "🆕 New Chat" button in the sidebar
2. A new session will be created automatically
3. Start typing your questions in the input field

### Managing Chat Sessions
- **View Previous Conversations**: All previous sessions are listed in the sidebar
- **Switch Sessions**: Click on any previous session to continue that conversation
- **Session Identification**: Each session shows a summary of the first message for easy identification

### Chat Interface
- Type your message in the input field at the bottom
- Click "Send" or press Enter to submit
- The AI will respond based on the conversation context
- All messages are preserved within each session

## 🏗️ Architecture

### Core Components

#### `QA_Chatbot.py`
- **Main Application**: Streamlit-based user interface
- **Session Management**: Handles creation and switching between chat sessions
- **Message Display**: Renders conversation history and real-time chat
- **Integration**: Connects with the History class for persistence

#### `Message_History.py`
- **History Class**: Core chat history management
- **Session Storage**: In-memory storage of chat sessions
- **Summary Generation**: Automatic summarization of conversations
- **Session Operations**: Create, access, and list chat sessions

### Key Classes and Methods

#### History Class
```python
class History:
    def __init__(self)                              # Initialize storage
    def existing_chat_access(session_id)            # Get chat history for session
    def new_chat()                                  # Create new chat session
    def chat_summary(session_id, message)           # Update session summary
    def summarize_message(message)                  # Generate message summary
    def list_sessions()                             # List all sessions with summaries
    def session_exists(session_id)                  # Check if session exists
    def get_session_count()                         # Get total session count
```

### Technology Stack

- **Framework**: Streamlit for web interface
- **LLM**: Google Gemini 1.5 Flash via LangChain
- **Chat History**: LangChain Community ChatMessageHistory
- **Session Management**: UUID-based session identification
- **Environment**: Python-dotenv for configuration

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini model | Yes |

### Model Configuration

The application uses Google's Gemini 1.5 Flash model by default. You can modify the model in `QA_Chatbot.py`:

```python
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
```

## 📊 Features in Detail

### Conversation History
- Each session maintains its own conversation context
- Messages are stored with type identification (human/assistant)
- History is preserved throughout the session lifetime
- Real-time display of conversation flow

### Session Management
- Unique UUID-based session identification
- Automatic session creation and management
- Session switching without data loss
- Summary-based session identification

### AI Integration
- Context-aware responses using conversation history
- System prompt for helpful assistant behavior
- Streaming response display with loading indicators
- Error handling for API failures

## 🛠️ Development

### Adding New Features

1. **Extend History Class**: Add new methods to `Message_History.py`
2. **Update UI**: Modify `QA_Chatbot.py` for interface changes
3. **Test Integration**: Ensure new features work with existing sessions

### Customizing the AI
- Modify the system prompt in `QA_Chatbot.py`
- Change the model by updating the `ChatGoogleGenerativeAI` initialization
- Add custom processing in the message chain

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure your `.env` file contains a valid `GOOGLE_API_KEY`
   - Check that the API key has proper permissions

2. **Session Not Found**:
   - The application will automatically create a new session
   - This typically happens on first run or after application restart

3. **Import Errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're in the correct virtual environment

### Debug Mode
To enable debug information, you can add logging to the application:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the code documentation
3. Create an issue in the repository

## 🔮 Future Enhancements

- [ ] Persistent storage (database integration)
- [ ] Export/import chat sessions
- [ ] Multiple AI model support
- [ ] Advanced session search and filtering
- [ ] Message editing and deletion
- [ ] Custom system prompts per session
- [ ] File upload support for context
- [ ] Authentication and user management

---

**Built with ❤️ using LangChain, Streamlit, and Google Gemini AI**
