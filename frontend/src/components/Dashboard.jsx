import React, { useMemo } from 'react';
import { ShieldAlert, TrendingDown, TrendingUp, Users, Download, Filter } from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, LineChart, Line
} from 'recharts';
import StudentTable from './StudentTable';

// Fallback mock data (used when no real data is uploaded)
const fallbackStudents = [
    { id: 1, name: "Alice Johnson", program: "Computer Science", attendance: 65, gpa: 2.1, score: 0.89, risk: "High", recommendations: ["Schedule 1-on-1 Mentoring", "Financial Aid Consultation"] },
    { id: 2, name: "Bob Smith", program: "Business Admin", attendance: 92, gpa: 3.4, score: 0.12, risk: "Low", recommendations: [] },
    { id: 3, name: "Charlie Davis", program: "Engineering", attendance: 75, gpa: 2.5, score: 0.55, risk: "Medium", recommendations: ["Send Early Warning Email", "Assign Peer Tutor"] },
    { id: 4, name: "Diana Prince", program: "Data Science", attendance: 88, gpa: 3.8, score: 0.05, risk: "Low", recommendations: [] },
    { id: 5, name: "Ethan Hunt", program: "Cybersecurity", attendance: 58, gpa: 1.9, score: 0.94, risk: "High", recommendations: ["Urgent Counselor Meeting", "Academic Advising"] },
];

const Dashboard = ({ dataLoaded, predictionData }) => {
    // Compute stats from real data if available
    const { students, riskDistribution, stats } = useMemo(() => {
        if (!predictionData || predictionData.length === 0) {
            return {
                students: fallbackStudents,
                riskDistribution: [
                    { name: 'Low Risk', value: 342, color: '#16a34a' },
                    { name: 'Medium Risk', value: 125, color: '#d97706' },
                    { name: 'High Risk', value: 48, color: '#dc2626' },
                ],
                stats: { total: 515, high: 48, avgAttendance: 72, modelAUC: 80.6 }
            };
        }

        // Transform real prediction data for display
        const studentList = predictionData.map((s, i) => ({
            id: s.Student_ID || i + 1,
            name: s.name || `Student ${s.Student_ID || i + 1}`,
            program: s.Department || 'N/A',
            attendance: Math.round(s.Attendance_Rate || s.attendance || 0),
            gpa: parseFloat(s.GPA || s.gpa || 0),
            score: parseFloat(s.risk_score || 0),
            risk: s.risk_level || 'Unknown',
            recommendations: s.risk_level === 'High'
                ? ['Schedule Counselor Meeting', 'Academic Support Review']
                : s.risk_level === 'Medium'
                    ? ['Monitor Progress', 'Assign Peer Mentor']
                    : [],
        }));

        // Sort by risk score descending
        studentList.sort((a, b) => parseFloat(b.score) - parseFloat(a.score));

        const highCount = studentList.filter(s => s.risk === 'High').length;
        const medCount = studentList.filter(s => s.risk === 'Medium').length;
        const lowCount = studentList.filter(s => s.risk === 'Low').length;

        const avgAtt = studentList.reduce((sum, s) => sum + s.attendance, 0) / studentList.length;

        return {
            students: studentList,
            riskDistribution: [
                { name: 'Low Risk', value: lowCount, color: '#16a34a' },
                { name: 'Medium Risk', value: medCount, color: '#d97706' },
                { name: 'High Risk', value: highCount, color: '#dc2626' },
            ],
            stats: {
                total: studentList.length,
                high: highCount,
                avgAttendance: Math.round(avgAtt),
                modelAUC: 80.6, // from trained model metrics
            }
        };
    }, [predictionData]);

    if (!dataLoaded) {
        return (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
                <div style={{ fontSize: '4rem', marginBottom: '1rem', opacity: 0.3 }}>📊</div>
                <h2 style={{ marginTop: '1rem', color: 'var(--text-primary)', fontSize: '1.5rem' }}>No Dataset Uploaded</h2>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Navigate to Upload Data to run the DropGuard AI analytics engine.</p>
            </div>
        );
    }

    const highRiskStudents = students.filter(s => s.risk === 'High' || s.risk === 'Medium');

    return (
        <>
            <div className="header">
                <div className="header-title">
                    <h1>Overview Dashboard</h1>
                    <p>Student Performance & Retention Analytics across {stats.total} students</p>
                </div>
                <div className="header-actions">
                    <button className="btn btn-secondary">
                        <Filter size={18} /> Filter
                    </button>
                    <button className="btn btn-primary">
                        <Download size={18} /> Export Report
                    </button>
                </div>
            </div>

            {stats.high > 0 && (
                <div className="alert-box">
                    <ShieldAlert className="alert-icon" size={20} />
                    <div className="alert-content">
                        <h4>System Alert: {stats.high} Students at High Risk</h4>
                        <p>The DropGuard v3 model identified {stats.high} students with high dropout probability. Immediate intervention recommended for priority cases below.</p>
                    </div>
                </div>
            )}

            <div className="dashboard-grid">
                <div className="stat-card">
                    <div className="stat-info">
                        <h3>High Risk Students</h3>
                        <div className="value" style={{ color: 'var(--risk-high)' }}>{stats.high}</div>
                        <div className="trend negative">
                            <ShieldAlert size={16} /> Requires immediate attention
                        </div>
                    </div>
                    <div className="icon-wrapper red">
                        <ShieldAlert size={24} />
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-info">
                        <h3>Avg Attendance</h3>
                        <div className="value">{stats.avgAttendance}%</div>
                        <div className="trend positive">
                            <Users size={16} /> Across {stats.total} students
                        </div>
                    </div>
                    <div className="icon-wrapper blue">
                        <Users size={24} />
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-info">
                        <h3>Model AUC-ROC</h3>
                        <div className="value" style={{ color: 'var(--risk-low)' }}>{stats.modelAUC}%</div>
                        <div className="trend positive">
                            <TrendingUp size={16} /> v3 Ensemble Model
                        </div>
                    </div>
                    <div className="icon-wrapper purple">
                        <TrendingUp size={24} />
                    </div>
                </div>
            </div>

            <div className="charts-grid">
                <div className="panel">
                    <div className="panel-header">
                        <div className="panel-title">Risk Score Distribution</div>
                    </div>
                    <div style={{ height: '280px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={students.slice(0, 20)}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
                                <XAxis dataKey="name" stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} angle={-45} textAnchor="end" height={80} />
                                <YAxis stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} domain={[0, 1]} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#fff', borderColor: '#e5e7eb', borderRadius: '6px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                                    formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Risk Score']}
                                />
                                <Bar dataKey="score" name="Risk Score" radius={[4, 4, 0, 0]}>
                                    {students.slice(0, 20).map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={entry.risk === 'High' ? '#dc2626' : entry.risk === 'Medium' ? '#d97706' : '#16a34a'}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="panel">
                    <div className="panel-header">
                        <div className="panel-title">Current Risk Distribution</div>
                    </div>
                    <div style={{ height: '280px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={riskDistribution}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={65}
                                    outerRadius={90}
                                    paddingAngle={2}
                                    dataKey="value"
                                    label={({ name, value }) => `${name}: ${value}`}
                                >
                                    {riskDistribution.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#fff', borderColor: '#e5e7eb', borderRadius: '6px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                                    itemStyle={{ color: '#111827' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="panel">
                <div className="panel-header">
                    <div className="panel-title">At-Risk Students Requiring Action ({highRiskStudents.length} students)</div>
                    <button className="btn btn-secondary">View All</button>
                </div>
                <StudentTable students={highRiskStudents.length > 0 ? highRiskStudents : students.slice(0, 5)} />
            </div>
        </>
    );
};

export default Dashboard;
