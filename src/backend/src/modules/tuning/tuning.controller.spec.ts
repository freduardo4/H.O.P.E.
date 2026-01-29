import { Test, TestingModule } from '@nestjs/testing';
import { TuningController } from './tuning.controller';
import { TuningService } from './tuning.service';
import { AfrService } from './afr.service';
import { OptimizeRequestDto, OptimizeResponseDto } from './tuning.dto';

describe('TuningController', () => {
    let controller: TuningController;
    let tuningService: TuningService;
    let afrService: AfrService;

    const mockTuningService = {
        optimizeMap: jest.fn().mockResolvedValue({ status: 'success', optimized_nodes: [] }),
    };

    const mockAfrService = {
        generateDefaultMap: jest.fn().mockReturnValue([[14.7, 14.7], [14.7, 14.7]]),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [TuningController],
            providers: [
                {
                    provide: TuningService,
                    useValue: mockTuningService,
                },
                {
                    provide: AfrService,
                    useValue: mockAfrService,
                },
            ],
        }).compile();

        controller = module.get<TuningController>(TuningController);
        tuningService = module.get<TuningService>(TuningService);
        afrService = module.get<AfrService>(AfrService);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    describe('optimize', () => {
        it('should call tuningService.optimizeMap with correct dto', async () => {
            const dto: OptimizeRequestDto = {
                current_map: [[14.7]],
                target_map: [[12.5]]
            };
            await controller.optimize(dto);
            expect(tuningService.optimizeMap).toHaveBeenCalledWith(dto);
        });
    });

    describe('getAfrTargets', () => {
        it('should call afrService.generateDefaultMap with default dimensions', () => {
            controller.getAfrTargets();
            expect(afrService.generateDefaultMap).toHaveBeenCalledWith(16, 16);
        });

        it('should call afrService.generateDefaultMap with custom dimensions', () => {
            controller.getAfrTargets(8, 8);
            expect(afrService.generateDefaultMap).toHaveBeenCalledWith(8, 8);
        });
    });
});
