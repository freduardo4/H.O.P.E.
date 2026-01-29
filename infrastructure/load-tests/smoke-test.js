import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    vus: 1, // 1 Virtual User
    duration: '10s', // Short duration
    thresholds: {
        http_req_failed: ['rate<0.01'], // http errors should be less than 1%
        http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    },
};

const BASE_URL = __ENV.BASE_URL || 'http://host.docker.internal:3000';

export default function () {
    // 1. Health Check
    const res = http.get(`${BASE_URL}/api/health`);
    check(res, {
        'status is 200': (r) => r.status === 200,
    });

    sleep(1);
}
