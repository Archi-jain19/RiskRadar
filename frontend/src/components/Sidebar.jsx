import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, UploadCloud, Settings, GraduationCap, Bell, Bot } from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab }) => {
    const menuItems = [
        { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={20} />, path: '/dashboard' },
        { id: 'students', label: 'Students', icon: <Users size={20} />, path: '/students' },
        { id: 'upload', label: 'Upload Data', icon: <UploadCloud size={20} />, path: '/upload' },
        { id: 'alerts', label: 'Interventions', icon: <Bell size={20} />, path: '/dashboard' },
        { id: 'settings', label: 'Settings', icon: <Settings size={20} />, path: '/settings' },
    ];

    return (
        <aside className="sidebar">
            <div className="logo">
                <GraduationCap size={28} color="var(--accent-blue)" />
                <span style={{ fontWeight: 700 }}>RiskRadar</span>
            </div>

            <nav>
                <ul className="nav-links">
                    {menuItems.map((item) => (
                        <li key={item.id}>
                            <NavLink
                                to={item.path}
                                className={({ isActive }) => `nav-item ${isActive && item.id === activeTab ? 'active' : ''}`}
                                onClick={() => setActiveTab(item.id)}
                            >
                                {item.icon}
                                {item.label}
                            </NavLink>
                        </li>
                    ))}
                    <li key="copilot-action" style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--panel-border)' }}>
                        <button
                            type="button"
                            className="nav-item"
                            style={{
                                width: '100%',
                                background: 'transparent',
                                border: 'none',
                                textAlign: 'left',
                                font: 'inherit',
                                cursor: 'pointer',
                                color: 'var(--accent-blue)',
                                fontWeight: 600,
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.75rem',
                                padding: '0.65rem 1rem'
                            }}
                            onClick={() => {
                                window.dispatchEvent(new CustomEvent('open-chatbot'));
                            }}
                        >
                            <Bot size={20} color="var(--accent-blue)" />
                            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                                AI Copilot
                                <span style={{ fontSize: '0.65rem', background: '#dbeafe', color: '#1d4ed8', padding: '2px 7px', borderRadius: '9999px', fontWeight: 700 }}>
                                    CHAT
                                </span>
                            </span>
                        </button>
                    </li>
                </ul>
            </nav>

            <div style={{ marginTop: 'auto', padding: '1rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', fontWeight: 500 }}>System Status</p>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--risk-low)' }}></div>
                    Analytics Engine Online
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
