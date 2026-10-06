import React from 'react';
import { Settings, Save } from 'lucide-react';

const SettingsPage = () => {
    return (
        <div style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
            <div className="header" style={{ marginBottom: '1.5rem' }}>
                <div className="header-title">
                    <h1>System Settings</h1>
                    <p>Configure your analytic preferences and notification settings</p>
                </div>
            </div>

            <div className="panel" style={{ marginBottom: '1.5rem', padding: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)' }}>
                    <Settings size={20} /> General Preferences
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--panel-border)', paddingBottom: '1.5rem' }}>
                        <div>
                            <h4 style={{ margin: '0 0 0.25rem 0', color: 'var(--text-primary)' }}>Email Notifications</h4>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Receive daily digests of at-risk students</p>
                        </div>
                        <label className="switch">
                            <input type="checkbox" defaultChecked />
                            <span className="slider round"></span>
                        </label>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--panel-border)', paddingBottom: '1.5rem' }}>
                        <div>
                            <h4 style={{ margin: '0 0 0.25rem 0', color: 'var(--text-primary)' }}>Risk Threshold Sensitivity</h4>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Adjust the ML confidence interval for flagging High Risk</p>
                        </div>
                        <select style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--panel-border)' }}>
                            <option>Conservative (Default)</option>
                            <option>Moderate</option>
                            <option>Aggressive (Flag Earlier)</option>
                        </select>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <h4 style={{ margin: '0 0 0.25rem 0', color: 'var(--text-primary)' }}>Automated Interventions</h4>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Automatically trigger generic LMS warning emails</p>
                        </div>
                        <label className="switch">
                            <input type="checkbox" />
                            <span className="slider round"></span>
                        </label>
                    </div>

                </div>

                <div style={{ marginTop: '2.5rem', display: 'flex', justifyContent: 'flex-end' }}>
                    <button className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Save size={18} /> Save Configurations
                    </button>
                </div>
            </div>

            <style>{`
                .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
                .switch input { opacity: 0; width: 0; height: 0; }
                .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; }
                .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .4s; }
                input:checked + .slider { background-color: var(--accent-blue); }
                input:checked + .slider:before { transform: translateX(20px); }
                .slider.round { border-radius: 24px; }
                .slider.round:before { border-radius: 50%; }
            `}</style>
        </div>
    );
};

export default SettingsPage;
