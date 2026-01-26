"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const testing_1 = require("@nestjs/testing");
const health_controller_1 = require("./health.controller");
describe('HealthController', () => {
    let controller;
    beforeEach(async () => {
        const module = await testing_1.Test.createTestingModule({
            controllers: [health_controller_1.HealthController],
        }).compile();
        controller = module.get(health_controller_1.HealthController);
    });
    it('should be defined', () => {
        expect(controller).toBeDefined();
    });
    describe('check()', () => {
        it('should return ok status', () => {
            const result = controller.check();
            expect(result.status).toBe('ok');
        });
        it('should return valid timestamp', () => {
            const result = controller.check();
            expect(new Date(result.timestamp).getTime()).not.toBeNaN();
        });
        it('should return uptime as a number', () => {
            const result = controller.check();
            expect(typeof result.uptime).toBe('number');
            expect(result.uptime).toBeGreaterThanOrEqual(0);
        });
        it('should return version string', () => {
            const result = controller.check();
            expect(typeof result.version).toBe('string');
            expect(result.version.length).toBeGreaterThan(0);
        });
    });
    describe('readiness()', () => {
        it('should return ready true', () => {
            const result = controller.readiness();
            expect(result.ready).toBe(true);
        });
    });
    describe('liveness()', () => {
        it('should return alive true', () => {
            const result = controller.liveness();
            expect(result.alive).toBe(true);
        });
    });
});
//# sourceMappingURL=health.controller.spec.js.map