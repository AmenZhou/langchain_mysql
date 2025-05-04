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
ENV PORT=4000

# Expose port 4000 for React dev server
EXPOSE 4000

# Start the React development server
CMD ["npm", "start"]

# Add test command
ENTRYPOINT ["npm", "test"]
