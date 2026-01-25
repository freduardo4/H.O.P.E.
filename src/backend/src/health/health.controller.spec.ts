import { Test, TestingModule } from '@nestjs/testing';
import { HealthController } from './health.controller';

describe('HealthController', () => {
    let controller: HealthController;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [HealthController],
        }).compile();

        controller = module.get<HealthController>(HealthController);
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
