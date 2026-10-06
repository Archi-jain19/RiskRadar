import React from 'react';
import { User, ShieldAlert, ArrowRight } from 'lucide-react';

const StudentTable = ({ students, setStudentData }) => {
    const [selectedStudent, setSelectedStudent] = React.useState(null);
    const [isEditing, setIsEditing] = React.useState(false);
    const [editData, setEditData] = React.useState({});
    const [isSaving, setIsSaving] = React.useState(false);

    const handleSelectStudent = (student) => {
        setSelectedStudent(student);
        setIsEditing(false);
        setEditData({
            gpa: student.gpa || 0,
            attendance: student.attendance || 0,
            program: student.program || '',
        });
    };

    const handleSave = async () => {
        setIsSaving(true);
        try {
            const res = await fetch(`http://127.0.0.1:8000/api/students/${selectedStudent.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gpa: parseFloat(editData.gpa),
                    attendance: parseFloat(editData.attendance),
                    program: editData.program
                })
            });
            const data = await res.json();
            if (res.ok) {
                if (setStudentData) {
                    setStudentData(prev => prev.map(s => s.id === selectedStudent.id ? { ...s, ...data.data, risk_level: s.risk_level, risk_score: s.risk_score } : s));
                }
                setSelectedStudent({ ...selectedStudent, ...data.data });
                setIsEditing(false);
            } else {
                alert('Failed to update student: ' + (data.detail || data.message));
            }
        } catch (err) {
            console.error(err);
            alert('Network error while saving');
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Student</th>
                        <th>Programs</th>
                        <th>Attendance %</th>
                        <th>GPA</th>
                        <th>Risk Score</th>
                        <th>Status</th>
                        <th>Recommended Intervention</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {students.map((student) => (
                        <tr key={student.id}>
                            <td>
                                <div className="student-info">
                                    <div className="avatar">{student.name.charAt(0)}</div>
                                    <div style={{ fontWeight: 600 }}>{student.name}</div>
                                </div>
                            </td>
                            <td>{student.program}</td>
                            <td>{student.attendance}%</td>
                            <td>{(Number(student.gpa) || 0).toFixed(2)}</td>
                            <td style={{ fontWeight: '500' }}>{(Number(student.score) || 0).toFixed(2)}</td>
                            <td>
                                <span className={`badge badge-${student.risk.toLowerCase()}`}>
                                    {student.risk === 'High' && <ShieldAlert size={14} />}
                                    {student.risk} Risk
                                </span>
                            </td>
                            <td style={{ color: 'var(--text-secondary)' }}>
                                {student.recommendations[0] || 'N/A'}
                            </td>
                            <td>
                                <button
                                    className="action-btn"
                                    title="View / Edit Details"
                                    onClick={() => handleSelectStudent(student)}
                                >
                                    <ArrowRight size={18} />
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {/* Simple Modal Overlay */}
            {selectedStudent && (
                <div style={{
                    position: 'fixed',
                    top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 999
                }} onClick={() => setSelectedStudent(null)}>
                    <div style={{
                        background: 'var(--panel-bg)',
                        padding: '2rem',
                        borderRadius: '12px',
                        width: '400px',
                        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                        position: 'relative'
                    }} onClick={e => e.stopPropagation()}>
                        <button
                            onClick={() => setSelectedStudent(null)}
                            style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.25rem', color: 'var(--text-secondary)' }}
                        >✕</button>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                <div className="avatar" style={{ width: '48px', height: '48px', fontSize: '1.25rem' }}>
                                    {selectedStudent.name.charAt(0)}
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                    <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)', margin: 0 }}>{selectedStudent.name}</h3>
                                    {isEditing ? (
                                        <input
                                            value={editData.program}
                                            onChange={e => setEditData({ ...editData, program: e.target.value })}
                                            style={{ padding: '4px', fontSize: '0.85rem', border: '1px solid var(--panel-border)', borderRadius: '4px' }}
                                        />
                                    ) : (
                                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', margin: 0 }}>{selectedStudent.program}</p>
                                    )}
                                </div>
                            </div>
                            <div>
                                {!isEditing ? (
                                    <button className="btn btn-secondary" onClick={() => setIsEditing(true)} style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>Edit</button>
                                ) : (
                                    <button className="btn btn-primary" onClick={handleSave} disabled={isSaving} style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
                                        {isSaving ? 'Saving...' : 'Save'}
                                    </button>
                                )}
                            </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                            <div style={{ background: 'var(--bg-color)', padding: '1rem', borderRadius: '8px' }}>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>GPA</p>
                                {isEditing ? (
                                    <input
                                        type="number" step="0.1"
                                        value={editData.gpa}
                                        onChange={e => setEditData({ ...editData, gpa: e.target.value })}
                                        style={{ width: '100%', padding: '4px', border: '1px solid var(--panel-border)', borderRadius: '4px' }}
                                    />
                                ) : (
                                    <p style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{(Number(selectedStudent.gpa) || 0).toFixed(2)}</p>
                                )}
                            </div>
                            <div style={{ background: 'var(--bg-color)', padding: '1rem', borderRadius: '8px' }}>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Attendance</p>
                                {isEditing ? (
                                    <input
                                        type="number"
                                        value={editData.attendance}
                                        onChange={e => setEditData({ ...editData, attendance: e.target.value })}
                                        style={{ width: '100%', padding: '4px', border: '1px solid var(--panel-border)', borderRadius: '4px' }}
                                    />
                                ) : (
                                    <p style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{selectedStudent.attendance}%</p>
                                )}
                            </div>
                        </div>

                        <div style={{ background: selectedStudent.risk === 'High' ? '#fee2e2' : selectedStudent.risk === 'Medium' ? '#fef3c7' : '#dcfce7', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
                            <p style={{ fontSize: '0.85rem', color: selectedStudent.risk === 'High' ? 'var(--risk-high)' : selectedStudent.risk === 'Medium' ? 'var(--risk-medium)' : 'var(--risk-low)', fontWeight: '600', marginBottom: '0.25rem' }}>Risk Score: {(Number(selectedStudent.score) || 0).toFixed(3)}</p>
                            <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)' }}>Assessment: <strong>{selectedStudent.risk}</strong></p>
                        </div>

                        <div>
                            <p style={{ fontSize: '0.9rem', fontWeight: '500', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Suggested Interventions</p>
                            <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.5' }}>
                                {selectedStudent.recommendations.length > 0 ? (
                                    selectedStudent.recommendations.map((rec, i) => <li key={i}>{rec}</li>)
                                ) : (
                                    <li>No immediate action required. Monitor standard progress.</li>
                                )}
                            </ul>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default StudentTable;
