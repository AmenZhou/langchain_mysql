const response = await axios.post(BACKEND_URL, {
  query: userMessage,
  response_type: "all"
}); 