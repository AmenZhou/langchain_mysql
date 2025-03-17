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
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
        question: userMessage,
      });

      // Extracting and format AI response
      const aiMessage = response.data.result.result.replace(/\n/g, "<br/>"); // Convert newlines to HTML

      // Displaying response to chat
      setMessages((prev) => [
        ...prev,
        { text: aiMessage, sender: "bot", isHTML: true },
      ]);
    } catch (error) {
      console.error("Error fetching AI response:", error);
      setMessages((prev) => [
        ...prev,
        { text: "❌ Error connecting to AI server.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
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
