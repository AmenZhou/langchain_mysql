# Use an official Node.js runtime image
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /frontend

COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Set environment variable for the port
ENV PORT=3000

# Set environment variable for backend API URL
ENV REACT_APP_BACKEND_URL=http://localhost:8000/query

# Expose port 3000 for React dev server
EXPOSE 3000

# Start the React development server
CMD ["npm", "start"]
