import React from 'react';

const MetricCard = ({ label, value, unit = '', isAlert = false }) => {
    return (
        <div className="metric-card">
            <div className="metric-label">{label}</div>
            <div className={`metric-value ${isAlert ? 'surprise' : ''}`}>
                {value}{unit}
            </div>
        </div>
    );
};

export default MetricCard;
