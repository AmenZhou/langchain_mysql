# Use an official Node.js runtime image
FROM node:16-alpine

# Set working directory inside the container
WORKDIR /frontend

COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose port 3000 for React dev server
EXPOSE 3000

# Start the React development server
CMD ["npm", "start"]
