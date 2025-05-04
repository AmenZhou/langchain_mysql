import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import ChatInput from '../ChatInput';

// Mock axios
jest.mock('axios');

describe('ChatInput', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  it('renders input field and send button', () => {
    render(<ChatInput />);
    
    // Check if input field is present
    const inputField = screen.getByPlaceholderText('Find the data you need...');
    expect(inputField).toBeInTheDocument();
    
    // Check if send button is present
    const sendButton = screen.getByRole('button');
    expect(sendButton).toBeInTheDocument();
  });

  it('sends message and displays response', async () => {
    // Mock axios response
    const mockResponse = {
      data: {
        result: {
          sql: 'SELECT * FROM users',
          data: [{ id: 1, name: 'John' }],
          explanation: 'This query returns all users'
        }
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    render(<ChatInput />);
    
    // Type a message
    const inputField = screen.getByPlaceholderText('Find the data you need...');
    fireEvent.change(inputField, { target: { value: 'Show me all users' } });
    
    // Click send button
    const sendButton = screen.getByRole('button');
    fireEvent.click(sendButton);
    
    // Wait for the response
    await waitFor(() => {
      const container = screen.getByTestId('chat-input');
      expect(container.innerHTML).toContain('SQL Query:');
      expect(container.innerHTML).toContain('SELECT * FROM users');
      expect(container.innerHTML).toContain('Results:');
      expect(container.innerHTML).toContain('This query returns all users');
    });
  });

  it('handles error response', async () => {
    // Mock axios error
    const mockError = new Error('Network error');
    mockError.response = {
      data: {
        detail: 'Error connecting to AI server.'
      }
    };
    axios.post.mockRejectedValueOnce(mockError);

    render(<ChatInput />);
    
    // Type a message
    const inputField = screen.getByPlaceholderText('Find the data you need...');
    fireEvent.change(inputField, { target: { value: 'Show me all users' } });
    
    // Click send button
    const sendButton = screen.getByRole('button');
    fireEvent.click(sendButton);
    
    // Wait for the error message
    await waitFor(() => {
      expect(screen.getByText('❌ Error connecting to AI server.')).toBeInTheDocument();
    });
  });
}); 
