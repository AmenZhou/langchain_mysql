import { render, screen } from '@testing-library/react';
import App from './App';

// Mock scrollIntoView
window.HTMLElement.prototype.scrollIntoView = function() {};

test('renders chat input component', () => {
  render(<App />);
  const chatInput = screen.getByTestId('chat-input');
  expect(chatInput).toBeInTheDocument();
}); 
