
import { Test, TestingModule } from '@nestjs/testing';
import { AfrService } from './afr.service';

describe('AfrService', () => {
    let service: AfrService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [AfrService],
        }).compile();

        service = module.get<AfrService>(AfrService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    it('should generate a 16x16 map by default', () => {
        const map = service.generateDefaultMap();
        expect(map.length).toBe(16);
        expect(map[0].length).toBe(16);
    });

    it('should generate stoichiometric AFR (14.7) for low load', () => {
        const map = service.generateDefaultMap();
        // Row 0 is lowest load
        const lowLoadRow = map[0];
        // Check various RPM points
        expect(lowLoadRow[0]).toBe(14.7); // Idle
        expect(lowLoadRow[15]).toBe(14.7); // Redline at low load
    });

    it('should generate rich AFR (< 13.0) for high load/RPM', () => {
        const map = service.generateDefaultMap();
        // Row 15 is highest load
        const highLoadRow = map[15];

        // At high RPM (last column), it should be rich
        const highRpmCell = highLoadRow[15];
        console.log('High Load/RPM AFR:', highRpmCell);
        expect(highRpmCell).toBeLessThan(13.0);

        // At low RPM but high load, it should still be somewhat rich or transitioning
        expect(highLoadRow[0]).toBeLessThan(14.7);
    });
});
