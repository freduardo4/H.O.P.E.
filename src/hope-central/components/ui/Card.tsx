import React from 'react';

interface CardProps {
    children: React.ReactNode;
    className?: string;
    hoverEffect?: boolean;
}

const Card: React.FC<CardProps> = ({
    children,
    className = '',
    hoverEffect = true
}) => {
    return (
        <div className={`
      bg-card border border-card-border rounded-2xl overflow-hidden
      ${hoverEffect ? 'hover:border-primary/50 transition-all group' : ''}
      ${className}
    `}>
            {children}
        </div>
    );
};

export const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
    <div className={`p-6 border-b border-card-border ${className}`}>{children}</div>
);

export const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
    <div className={`p-6 ${className}`}>{children}</div>
);

export const CardFooter: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
    <div className={`p-6 border-t border-card-border bg-black/20 ${className}`}>{children}</div>
);

export default Card;
