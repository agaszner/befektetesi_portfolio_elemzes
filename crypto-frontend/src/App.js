import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState('Loading prediction...');

  useEffect(() => {
    // Simulate an API call to fetch crypto price prediction
    setTimeout(() => {
      setPrediction("Bitcoin (BTC): $60,000 predicted for next week.");
    }, 1000);
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Crypto Price Prediction</h1>
      </header>
      <main>
        <section className="prediction-container">
          <h2>Latest Prediction</h2>
          <div>{prediction}</div>
        </section>
      </main>
    </div>
  );
}

export default App;
