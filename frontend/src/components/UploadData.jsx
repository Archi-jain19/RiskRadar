import React from 'react';
import { UploadCloud, FileSpreadsheet, CheckCircle, ArrowRight, Loader } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const UploadData = ({ setDataLoaded, setActiveTab, setStudentData }) => {
    const navigate = useNavigate();
    const [isUploading, setIsUploading] = React.useState(false);
    const fileInputRef = React.useRef(null);

    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('http://127.0.0.1:8000/api/upload-student-data', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (res.ok) {
                setStudentData(data.data);
                setDataLoaded(true);
                setActiveTab('dashboard');
                navigate('/dashboard');
            } else {
                alert('Upload failed: ' + (data.detail || data.message));
            }
        } catch (error) {
            console.error(error);
            alert('Failed to connect to the backend server.');
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    return (
        <>
            <div className="header">
                <div className="header-title">
                    <h1>Upload Dataset</h1>
                    <p>Import student records via CSV for retention analysis</p>
                </div>
            </div>

            <div className="panel" style={{ maxWidth: '800px', margin: '0 auto' }}>
                <div
                    className="upload-area"
                    onClick={handleUploadClick}
                    style={{ pointerEvents: isUploading ? 'none' : 'auto', opacity: isUploading ? 0.7 : 1 }}
                >
                    <input
                        type="file"
                        accept=".csv"
                        style={{ display: 'none' }}
                        ref={fileInputRef}
                        onChange={handleFileChange}
                    />
                    {isUploading ? (
                        <>
                            <div className="upload-icon" style={{ animation: 'spin 2s linear infinite' }}>
                                <Loader size={28} />
                            </div>
                            <div className="upload-text">
                                <h3>Analyzing Student Data...</h3>
                                <p>Evaluating academic and behavioral factors against historical metrics</p>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="upload-icon">
                                <UploadCloud size={28} />
                            </div>
                            <div className="upload-text">
                                <h3>Click to Upload Student Data</h3>
                                <p>Drag and drop or select a CSV/Excel file up to 50MB</p>
                                <button className="btn btn-secondary" style={{ marginTop: '1.5rem', pointerEvents: 'none' }}>
                                    <FileSpreadsheet size={16} /> Select CSV File
                                </button>
                            </div>
                        </>
                    )}
                </div>

                <div style={{ marginTop: '2.5rem' }}>
                    <h4 style={{ marginBottom: '1rem', color: 'var(--text-primary)', fontSize: '0.95rem' }}>Supported Fields Format</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        <div className="panel" style={{ background: '#f8fafc', padding: '1.25rem', boxShadow: 'none' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                <CheckCircle size={16} color="var(--risk-low)" />
                                <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.9rem' }}>Academic</span>
                            </div>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>GPA, Assignment Scores, Midterm Grades</p>
                        </div>
                        <div className="panel" style={{ background: '#f8fafc', padding: '1.25rem', boxShadow: 'none' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                <CheckCircle size={16} color="var(--risk-low)" />
                                <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.9rem' }}>Behavioral & Engagement</span>
                            </div>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Attendance %, LMS Logins, Library Visits</p>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default UploadData;
