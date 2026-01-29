import React from 'react';

interface StarRatingProps {
    rating: number;
    max?: number;
}

const StarRating: React.FC<StarRatingProps> = ({ rating, max = 5 }) => {
    return (
        <div className="flex items-center gap-0.5">
            {[...Array(max)].map((_, i) => (
                <svg
                    key={i}
                    xmlns="http://www.w3.org/2000/svg"
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill={i < Math.round(rating) ? "currentColor" : "none"}
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className={i < Math.round(rating) ? "text-secondary" : "text-gray-600"}
                >
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
            ))}
            <span className="ml-1.5 text-[10px] font-black text-gray-500">{rating.toFixed(1)}</span>
        </div>
    );
};

export default StarRating;
