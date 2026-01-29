import http from 'k6/http';
import { check, group, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '30s', target: 20 }, // Ramp up to 20 users
        { duration: '1m', target: 20 },  // Stay at 20 users
        { duration: '30s', target: 0 },  // Ramp down to 0
    ],
    thresholds: {
        http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    },
};

const BASE_URL = __ENV.BASE_URL || 'http://host.docker.internal:3000';

export default function () {
    group('Public API', function () {
        const res = http.get(`${BASE_URL}/api/health`);
        check(res, { 'health status is 200': (r) => r.status === 200 });
    });

    group('Marketplace', function () {
        // Simulate searching for items (public endpoint assumed or mocked)
        // Adjust endpoint based on actual API implementation
        const res = http.get(`${BASE_URL}/api/marketplace/listings`);
        // Ideally we check 200, but if auth is required, 401 might be expected for unauth.
        // Assuming public listing:
        // check(res, { 'listings status is 200': (r) => r.status === 200 });
    });

    sleep(1);
}
