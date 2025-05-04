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
      const response = await axios.post("http://localhost:8000/query", {
        query: userMessage,
      });

      // Handle the response based on its structure
      let aiMessage = "";
      if (response.data.result) {
        if (typeof response.data.result === 'object') {
          // If result is an object (for response_type="all")
          aiMessage = `
            <div class="response-container">
              <div class="sql-query">
                <strong>SQL Query:</strong><br/>
                <pre>${response.data.result.sql}</pre>
              </div>
              <div class="data">
                <strong>Results:</strong><br/>
                <pre>${JSON.stringify(response.data.result.data, null, 2)}</pre>
              </div>
              <div class="explanation">
                <strong>Explanation:</strong><br/>
                ${response.data.result.explanation}
              </div>
            </div>
          `;
        } else {
          // If result is a string (for other response types)
          aiMessage = response.data.result;
        }
      } else {
        // Fallback to individual fields if result is not present
        aiMessage = `
          <div class="response-container">
            ${response.data.sql ? `<div class="sql-query"><strong>SQL Query:</strong><br/><pre>${response.data.sql}</pre></div>` : ''}
            ${response.data.data ? `<div class="data"><strong>Results:</strong><br/><pre>${JSON.stringify(response.data.data, null, 2)}</pre></div>` : ''}
            ${response.data.explanation ? `<div class="explanation"><strong>Explanation:</strong><br/>${response.data.explanation}</div>` : ''}
          </div>
        `;
      }

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
