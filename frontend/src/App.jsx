import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Students from './components/Students';
import UploadData from './components/UploadData';
import SettingsPage from './components/SettingsPage';
import ChatbotButton from './components/ChatbotButton';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [dataLoaded, setDataLoaded] = useState(false);
  const [studentData, setStudentData] = useState([]);

  useEffect(() => {
    // Auto-fetch existing predictions from Supabase on load
    fetch('http://127.0.0.1:8000/api/student-predictions')
      .then(res => res.json())
      .then(result => {
        if (result.status === 'success' && result.data && result.data.length > 0) {
          setStudentData(result.data);
          setDataLoaded(true);
        }
      })
      .catch(err => console.log('Backend not yet populated or offline.'));
  }, []);

  return (
    <Router>
      <div className="app-container">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard dataLoaded={dataLoaded} predictionData={studentData} />} />
            <Route path="/students" element={<Students dataLoaded={dataLoaded} studentData={studentData} setStudentData={setStudentData} />} />
            <Route path="/upload" element={
              <UploadData
                setDataLoaded={setDataLoaded}
                setActiveTab={setActiveTab}
                setStudentData={setStudentData}
              />
            } />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Dashboard dataLoaded={dataLoaded} predictionData={studentData} />} />
          </Routes>
        </main>
        <ChatbotButton />
      </div>
    </Router>
  );
}

export default App;
