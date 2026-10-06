import React, { useState, useRef, useEffect } from 'react';
import { HelpCircle, X, Send, Bot, Sparkles } from 'lucide-react';

const ChatbotButton = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { text: "Hello! Welcome to the Faculty Support Desk. How can I assist you with analyzing student data or planning interventions today?", sender: 'bot' }
    ]);
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (isOpen) {
            scrollToBottom();
        }
    }, [messages, isOpen]);

    useEffect(() => {
        const handleOpen = () => setIsOpen(true);
        window.addEventListener('open-chatbot', handleOpen);
        return () => window.removeEventListener('open-chatbot', handleOpen);
    }, []);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = input;
        setMessages(prev => [...prev, { text: userMsg, sender: 'user' }]);
        setInput('');

        try {
            const res = await fetch('http://127.0.0.1:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg })
            });
            const data = await res.json();

            setMessages(prev => [...prev, {
                text: data.reply || "An error occurred while fetching the response.",
                sender: 'bot'
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                text: "Network error: Unable to reach the support server.",
                sender: 'bot'
            }]);
        }
    };

    return (
        <>
            <button
                className="chatbot-btn"
                onClick={() => setIsOpen(!isOpen)}
                title="Open AI Faculty Copilot"
                style={{
                    transform: isOpen ? 'scale(0)' : 'scale(1)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.6rem',
                    padding: '0.75rem 1.25rem',
                    width: 'auto',
                    height: 'auto',
                    borderRadius: '9999px',
                    background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                    boxShadow: '0 10px 25px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.1)',
                    border: '1px solid rgba(56, 189, 248, 0.3)',
                    color: '#ffffff',
                    cursor: 'pointer',
                    zIndex: 100
                }}
            >
                <Bot size={22} color="#38bdf8" />
                <span style={{ fontWeight: 600, fontSize: '0.9rem', letterSpacing: '0.01em' }}>Ask AI Copilot</span>
                <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: '#22c55e',
                    boxShadow: '0 0 8px #22c55e'
                }}></span>
            </button>

            {isOpen && (
                <div style={{
                    position: 'fixed',
                    bottom: '2rem',
                    right: '2rem',
                    width: '350px',
                    height: '480px',
                    background: '#ffffff',
                    border: '1px solid var(--panel-border)',
                    borderRadius: '12px',
                    display: 'flex',
                    flexDirection: 'column',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                    zIndex: 100,
                    overflow: 'hidden'
                }}>
                    <div style={{
                        padding: '1rem 1.25rem',
                        background: 'var(--text-primary)',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        color: 'white'
                    }}>
                        <h3 style={{ fontSize: '0.95rem', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <HelpCircle size={18} /> Faculty Support Desk
                        </h3>
                        <button
                            onClick={() => setIsOpen(false)}
                            style={{ background: 'transparent', border: 'none', color: '#e2e8f0', cursor: 'pointer' }}
                        >
                            <X size={20} />
                        </button>
                    </div>

                    <div style={{ flex: 1, padding: '1.25rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem', background: '#f8fafc' }}>
                        {messages.map((m, i) => (
                            <div key={i} style={{
                                alignSelf: m.sender === 'user' ? 'flex-end' : 'flex-start',
                                background: m.sender === 'user' ? 'var(--accent-blue)' : '#ffffff',
                                color: m.sender === 'user' ? '#ffffff' : 'var(--text-primary)',
                                maxWidth: '85%',
                                padding: '0.75rem 1rem',
                                borderRadius: '8px',
                                borderBottomLeftRadius: m.sender === 'bot' ? '0' : '8px',
                                borderBottomRightRadius: m.sender === 'user' ? '0' : '8px',
                                fontSize: '0.85rem',
                                lineHeight: 1.5,
                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                border: m.sender === 'bot' ? '1px solid #e2e8f0' : 'none'
                            }}>
                                <div dangerouslySetInnerHTML={{ __html: m.text }} />
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    <form onSubmit={handleSend} style={{ display: 'flex', borderTop: '1px solid var(--panel-border)', background: '#ffffff', padding: '0.5rem' }}>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type your message..."
                            style={{
                                flex: 1,
                                padding: '0.75rem',
                                background: 'transparent',
                                border: 'none',
                                color: 'var(--text-primary)',
                                outline: 'none',
                                fontSize: '0.9rem'
                            }}
                        />
                        <button type="submit" style={{
                            background: 'transparent',
                            border: 'none',
                            padding: '0 0.75rem',
                            color: 'var(--accent-blue)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <Send size={18} />
                        </button>
                    </form>
                </div>
            )}
        </>
    );
};

export default ChatbotButton;
