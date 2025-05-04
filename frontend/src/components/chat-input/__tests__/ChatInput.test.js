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

  it('displays response data correctly', async () => {
    // Mock axios response with the example data
    const mockResponse = {
      data: {
        result: {
          sql: "SELECT * FROM consult_statuses WHERE consultation_id = 1;",
          data: [
            {
              consult_status_id: 1,
              consultation_id: 1,
              consultee_person_id: 144,
              consultee_person_type: "Member",
              status_cd: "CONSULTSTATUS_REQ",
              consult_status_dt: "2025-04-14T19:57:58",
              reason_type: "MasterCode",
              reason_cd: "MBRREQUEST_TOVIDEO",
              data_source_cd: "TAS",
              exclusion_cd: "IN",
              created_at: "2025-04-14T19:57:58",
              created_by: 830,
              updated_at: "2025-04-14T19:57:58",
              updated_by: 830
            }
          ],
          explanation: "The SQL query is selecting all columns from the table..."
        },
        sql: "SELECT * FROM consult_statuses WHERE consultation_id = 1;",
        data: [
          {
            consult_status_id: 1,
            consultation_id: 1,
            consultee_person_id: 144,
            consultee_person_type: "Member",
            status_cd: "CONSULTSTATUS_REQ",
            consult_status_dt: "2025-04-14T19:57:58",
            reason_type: "MasterCode",
            reason_cd: "MBRREQUEST_TOVIDEO",
            data_source_cd: "TAS",
            exclusion_cd: "IN",
            created_at: "2025-04-14T19:57:58",
            created_by: 830,
            updated_at: "2025-04-14T19:57:58",
            updated_by: 830
          }
        ],
        explanation: "The SQL query is selecting all columns from the table...",
        response_type: "all"
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
    
    // Wait for the response and verify its display
    await waitFor(() => {
      // Check if SQL query is displayed
      expect(screen.getByText(/SQL Query:/i)).toBeInTheDocument();
      expect(screen.getByText(/SELECT \* FROM consult_statuses WHERE consultation_id = 1;/i)).toBeInTheDocument();
      
      // Check if data is displayed
      expect(screen.getByText(/Results:/i)).toBeInTheDocument();
      expect(screen.getByText(/"consult_status_id": 1/i)).toBeInTheDocument();
      expect(screen.getByText(/"consultee_person_type": "Member"/i)).toBeInTheDocument();
      
      // Check if explanation is displayed
      expect(screen.getByText(/Explanation:/i)).toBeInTheDocument();
      expect(screen.getByText(/The SQL query is selecting all columns from the table/i)).toBeInTheDocument();
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
