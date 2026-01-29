import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '1m', target: 100 }, // Ramp up to 100 users
        { duration: '2m', target: 100 }, // Stay at 100 users
        { duration: '1m', target: 0 },   // Ramp down
    ],
};

const BASE_URL = __ENV.BASE_URL || 'http://host.docker.internal:3000';

export default function () {
    // Hit a potentially expensive endpoint
    http.get(`${BASE_URL}/api/health`);
    sleep(0.5);
}
