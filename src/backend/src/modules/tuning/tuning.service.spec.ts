
import { Test, TestingModule } from '@nestjs/testing';
import { TuningService } from './tuning.service';
import { OptimizeRequestDto } from './tuning.dto';

describe('TuningService', () => {
    let service: TuningService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [TuningService],
        }).compile();

        service = module.get<TuningService>(TuningService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    it('should optimize a simple map using python script', async () => {
        const request: OptimizeRequestDto = {
            current_map: [
                [5, 5],
                [5, 5]
            ],
            target_map: [
                [10, 10],
                [10, 10]
            ],
            config: {
                generations: 5,
                population_size: 10,
                mutation_strength: 0.3
            }
        };

        try {
            const result = await service.optimizeMap(request);
            expect(result).toBeDefined();
            expect(result.status).toBe('success');
            expect(result.optimized_map).toBeDefined();
            expect(result.fitness_history).toBeDefined();
            expect(result.optimized_map.length).toBe(2);

            // Check if it moved towards 10 (it might not reach it in 5 gens, but should change)
            const centerVal = result.optimized_map[0][0];
            console.log('Optimized value:', centerVal);
            // It should defineitaly be different from 5
            expect(centerVal).not.toBe(5);
        } catch (e) {
            console.error('Optimization test failed:', e);
            throw e;
        }
    }, 30000); // increase timeout for python startup
});
