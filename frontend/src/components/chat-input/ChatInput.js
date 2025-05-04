import React, { useState, useEffect, useRef } from "react";
import "./styles.css";
import SendIcon from "../../icons/SendIcon";
import axios from "axios";

const ChatInput = () => {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I assist you today?", sender: "bot" },
  ]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Scroll to bottom when messages update
  useEffect(() => {
    if (chatEndRef.current && typeof chatEndRef.current.scrollIntoView === 'function') {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSend = async () => {
    if (message.trim() === "") return;

    // Add user message to chat
    setMessages((prev) => [...prev, { text: message, sender: "user" }]);
    const userMessage = message;
    setMessage(""); // Clear input
    setLoading(true);

    try {
      // Sending message to FastAPI backend
      const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000/query";
      const response = await axios.post(BACKEND_URL, {
        query: userMessage,
      });

      // Display raw response for debugging
      console.log("Backend response:", JSON.stringify(response.data, null, 2));

      // Extract the SQL, data, and explanation while safely handling missing fields
      let sqlQuery = "";
      let resultData = [];
      let explanationText = "";

      // First try to get data from top-level fields
      if (typeof response.data === 'object' && response.data !== null) {
        if (typeof response.data.sql === 'string') {
          sqlQuery = response.data.sql;
        }
        
        if (Array.isArray(response.data.data)) {
          resultData = response.data.data;
        }
        
        if (typeof response.data.explanation === 'string') {
          explanationText = response.data.explanation;
        }
        
        // If any fields are missing, try to get them from the result object
        if (response.data.result && typeof response.data.result === 'object') {
          if (!sqlQuery && typeof response.data.result.sql === 'string') {
            sqlQuery = response.data.result.sql;
          }
          
          if (resultData.length === 0 && Array.isArray(response.data.result.data)) {
            resultData = response.data.result.data;
          }
          
          if (!explanationText && typeof response.data.result.explanation === 'string') {
            explanationText = response.data.result.explanation;
          }
        }
      }

      // Build the HTML response
      const aiMessage = `
        <div class="response-container">
          <div class="sql-query">
            <strong>SQL Query:</strong><br/>
            <pre>${sqlQuery || 'No SQL query generated'}</pre>
          </div>
          <div class="data">
            <strong>Results:</strong><br/>
            <pre>${JSON.stringify(resultData || [], null, 2)}</pre>
          </div>
          <div class="explanation">
            <strong>Explanation:</strong><br/>
            ${explanationText || 'No explanation available'}
          </div>
        </div>
      `;

      // Displaying response to chat
      setMessages((prev) => [
        ...prev,
        { text: aiMessage, sender: "bot", isHTML: true },
      ]);
    } catch (error) {
      console.error("Error fetching AI response:", error);
      let errorMessage = "❌ Error connecting to AI server.";
      
      // Check if the error has a response from the backend
      if (error.response && error.response.data && error.response.data.detail) {
        errorMessage = `❌ ${error.response.data.detail}`;
      }
      
      setMessages((prev) => [
        ...prev,
        { text: errorMessage, sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container" data-testid="chat-input">
      {/* Chat Window */}
      <h1 className="title">AI Powered DB Assistant</h1>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {msg.isHTML ? (
              <span dangerouslySetInnerHTML={{ __html: msg.text }} />
            ) : (
              msg.text
            )}
          </div>
        ))}
        {loading && <div className="message bot">DB Assistant is typing...</div>}
        <div ref={chatEndRef} />
      </div>

      {/* Chat Input */}
      <div className="chat-input-container">
        <input
          type="text"
          className="chat-input"
          placeholder="Find the data you need..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={loading}
        >
          <SendIcon className="chat-send-icon" size={20} />
        </button>
      </div>
    </div>
  );
};

export default ChatInput;
