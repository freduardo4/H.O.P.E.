import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from './config.service';

describe('ConfigService', () => {
    let service: ConfigService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [ConfigService],
        }).compile();

        service = module.get<ConfigService>(ConfigService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    describe('getFlags', () => {
        it('should return all flags', () => {
            const flags = service.getFlags();
            expect(flags).toBeDefined();
            expect(flags).toHaveProperty('enableBetaFlashing');
            expect(flags).toHaveProperty('minRequiredVoltage');
        });
    });

    describe('getFlag', () => {
        it('should return a specific flag value', () => {
            const maintenanceMode = service.getFlag('maintenanceMode');
            expect(typeof maintenanceMode).toBe('boolean');

            const minVoltage = service.getFlag('minRequiredVoltage');
            expect(minVoltage).toBe(12.5);
        });
    });
});
