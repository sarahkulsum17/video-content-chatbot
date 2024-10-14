import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [video, setVideo] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [summary, setSummary] = useState('');

  const handleVideoUpload = (e) => {
    setVideo(e.target.files[0]);
  };

  const handleSubmitQuestion = async () => {
    const formData = new FormData();
    formData.append('video', video);

    await axios.post('http://54.81.64.39:3000/upload', formData);

    const response = await axios.post('http://54.81.64.39:3000/chatbot', {
      question,
      summarize: false,
    });

    setAnswer(response.data.answer);
  };

  const handleGetSummary = async () => {
    const response = await axios.post('http://54.81.64.39:3000/chatbot', {
      summarize: true,
    });

    setSummary(response.data.summary);
  };

  return (
    <div>
      <h1>Video Query System with AI Chatbot</h1>
      <input type="file" onChange={handleVideoUpload} />
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question"
      />
      <button onClick={handleSubmitQuestion}>Submit Question</button>
      <button onClick={handleGetSummary}>Get Video Summary</button>

      <h3>Answer: {answer}</h3>
      <h3>Summary: {summary}</h3>
    </div>
  );
}

export default App;




