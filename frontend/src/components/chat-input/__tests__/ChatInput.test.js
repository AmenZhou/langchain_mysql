import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import axios from 'axios';
import ChatInput from '../ChatInput';

// Mock axios
jest.mock('axios');

// Mock scrollIntoView
window.HTMLElement.prototype.scrollIntoView = function() {};

describe('ChatInput Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  it('renders initial bot message', () => {
    render(<ChatInput />);
    expect(screen.getByText('Hello! How can I assist you today?')).toBeInTheDocument();
  });

  it('sends message when enter key is pressed', async () => {
    // Mock successful API response
    const mockResponse = { data: { result: 'Test response' } };
    axios.post.mockImplementation(() => new Promise((resolve) => {
      setTimeout(() => resolve(mockResponse), 100);
    }));

    render(<ChatInput />);
    const input = screen.getByPlaceholderText('Find the data you need...');
    
    // Type a message and press enter
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Test message' } });
      fireEvent.keyDown(input, { key: 'Enter' });
    });

    // Check if loading state is shown
    expect(screen.getByText('DB Assistant is typing...')).toBeInTheDocument();

    // Wait for the API call to complete and response to be displayed
    await waitFor(() => {
      expect(screen.getByText('Test response')).toBeInTheDocument();
    });

    // Verify API call
    expect(axios.post).toHaveBeenCalledWith('http://localhost:8000/query', {
      question: 'Test message'
    });
  });

  it('handles API errors gracefully', async () => {
    // Mock API error
    const mockError = {
      response: {
        data: {
          detail: 'Test error message'
        }
      }
    };
    axios.post.mockImplementation(() => new Promise((_, reject) => {
      setTimeout(() => reject(mockError), 100);
    }));

    render(<ChatInput />);
    const input = screen.getByPlaceholderText('Find the data you need...');
    
    // Type a message and press enter
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Test message' } });
      fireEvent.keyDown(input, { key: 'Enter' });
    });

    // Wait for the error message to appear
    await waitFor(() => {
      expect(screen.getByText('❌ Test error message')).toBeInTheDocument();
    });
  });

  it('does not send empty messages', () => {
    render(<ChatInput />);
    const input = screen.getByPlaceholderText('Find the data you need...');
    
    // Try to send an empty message
    fireEvent.change(input, { target: { value: '' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    // Check that no API call was made
    expect(axios.post).not.toHaveBeenCalled();
  });

  it('displays nonexistent table error from backend', async () => {
    // Mock API error for nonexistent table
    const mockError = {
      response: {
        status: 400,
        data: {
          detail: 'Database Error: Table "nonexistent_table" does not exist'
        }
      }
    };
    axios.post.mockImplementation(() => new Promise((_, reject) => {
      setTimeout(() => reject(mockError), 100);
    }));

    render(<ChatInput />);
    const input = screen.getByPlaceholderText('Find the data you need...');
    
    // Type a message and press enter
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Show me data from nonexistent_table' } });
      fireEvent.keyDown(input, { key: 'Enter' });
    });

    // Wait for the error message to appear
    await waitFor(() => {
      expect(screen.getByText('❌ Database Error: Table "nonexistent_table" does not exist')).toBeInTheDocument();
    });
  });

  it('displays rate limit error from backend', async () => {
    // Mock API error for rate limit
    const mockError = {
      response: {
        status: 429,
        data: {
          detail: 'Rate limit exceeded. Please try again later.'
        }
      }
    };
    axios.post.mockImplementation(() => new Promise((_, reject) => {
      setTimeout(() => reject(mockError), 100);
    }));

    render(<ChatInput />);
    const input = screen.getByPlaceholderText('Find the data you need...');
    
    // Type a message and press enter
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Any question' } });
      fireEvent.keyDown(input, { key: 'Enter' });
    });

    // Wait for the error message to appear
    await waitFor(() => {
      expect(screen.getByText('❌ Rate limit exceeded. Please try again later.')).toBeInTheDocument();
    });
  });

  it('displays general database error from backend', async () => {
    // Mock API error for general database error
    const mockError = {
      response: {
        status: 500,
        data: {
          detail: 'An unexpected database error occurred'
        }
      }
    };
    axios.post.mockImplementation(() => new Promise((_, reject) => {
      setTimeout(() => reject(mockError), 100);
    }));

    render(<ChatInput />);
    const input = screen.getByPlaceholderText('Find the data you need...');
    
    // Type a message and press enter
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Any question' } });
      fireEvent.keyDown(input, { key: 'Enter' });
    });

    // Wait for the error message to appear
    await waitFor(() => {
      expect(screen.getByText('❌ An unexpected database error occurred')).toBeInTheDocument();
    });
  });
}); 
