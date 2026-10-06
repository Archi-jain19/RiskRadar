import React, { useState } from 'react';
import StudentTable from './StudentTable';
import { Search } from 'lucide-react';

const Students = ({ dataLoaded, studentData, setStudentData }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [filterRisk, setFilterRisk] = useState('All');

    if (!dataLoaded || !studentData || studentData.length === 0) {
        return (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
                <img src="https://cdni.iconscout.com/illustration/premium/thumb/empty-state-2130362-1800926.png" alt="No data" style={{ width: '250px', opacity: 0.8 }} />
                <h2 style={{ marginTop: '2rem', color: 'var(--text-primary)', fontSize: '1.5rem' }}>No Students Found</h2>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Navigate to Upload Data to populate the directory.</p>
            </div>
        );
    }

    let formattedStudents = studentData.map((s, idx) => ({
        id: s.id || idx + 1,
        name: s.name || s.student_id || `Student #${idx + 1}`,
        program: s.program || 'Unknown',
        attendance: s.attendance || 0,
        gpa: s.gpa || 0.0,
        score: s.risk_score || 0.0,
        risk: s.risk_level || 'Low',
        recommendations: s.risk_level === 'High' ? ["Schedule 1-on-1 Mentoring"] : []
    })).sort((a, b) => b.score - a.score);

    if (filterRisk !== 'All') {
        formattedStudents = formattedStudents.filter(s => s.risk === filterRisk);
    }

    if (searchTerm) {
        formattedStudents = formattedStudents.filter(s =>
            s.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            s.program.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }

    return (
        <>
            <div className="header" style={{ marginBottom: '1.5rem' }}>
                <div className="header-title">
                    <h1>Student Directory</h1>
                    <p>Complete list of all enrolled students and their current status</p>
                </div>
            </div>

            <div className="panel" style={{ marginBottom: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'center', padding: '1rem 1.5rem' }}>
                <div style={{ flex: 1, position: 'relative' }}>
                    <Search size={18} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                    <input
                        type="text"
                        placeholder="Search students by name or program..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        style={{ width: '100%', padding: '0.65rem 1rem 0.65rem 2.75rem', borderRadius: '6px', border: '1px solid var(--panel-border)', outline: 'none', fontSize: '0.95rem' }}
                    />
                </div>
                <select
                    value={filterRisk}
                    onChange={(e) => setFilterRisk(e.target.value)}
                    style={{ padding: '0.65rem 1rem', borderRadius: '6px', border: '1px solid var(--panel-border)', outline: 'none', background: '#fff', fontSize: '0.95rem', color: 'var(--text-primary)', cursor: 'pointer' }}
                >
                    <option value="All">All Risk Levels</option>
                    <option value="High">High Risk</option>
                    <option value="Medium">Medium Risk</option>
                    <option value="Low">Low Risk</option>
                </select>
            </div>

            <div className="panel" style={{ padding: '0', overflow: 'hidden' }}>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--panel-border)' }}>
                    <div className="panel-title">Active Students ({formattedStudents.length})</div>
                </div>
                <div style={{ padding: '0 1.5rem 1.5rem 1.5rem' }}>
                    <StudentTable students={formattedStudents} setStudentData={setStudentData} />
                </div>
            </div>
        </>
    );
};

export default Students;
