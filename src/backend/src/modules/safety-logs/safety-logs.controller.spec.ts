import { Test, TestingModule } from '@nestjs/testing';
import { SafetyLogsController } from './safety-logs.controller';
import { SafetyLogsService } from './safety-logs.service';
import { ValidateFlashDto } from './dto/validate-flash.dto';
import { SafetyEventDto } from './dto/safety-event.dto';

describe('SafetyLogsController', () => {
    let controller: SafetyLogsController;
    let service: SafetyLogsService;

    const mockSafetyLogsService = {
        validateFlash: jest.fn().mockResolvedValue({ allowed: true }),
        logEvent: jest.fn().mockResolvedValue({ id: 1, type: 'FLASH_START' }),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            controllers: [SafetyLogsController],
            providers: [
                {
                    provide: SafetyLogsService,
                    useValue: mockSafetyLogsService,
                },
            ],
        }).compile();

        controller = module.get<SafetyLogsController>(SafetyLogsController);
        service = module.get<SafetyLogsService>(SafetyLogsService);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    describe('validateFlash', () => {
        it('should call service.validateFlash with correct dto', async () => {
            const dto: ValidateFlashDto = { ecuId: 'ECU123', voltage: 13.5 };
            await controller.validateFlash(dto);
            expect(service.validateFlash).toHaveBeenCalledWith(dto);
        });
    });

    describe('logEvent', () => {
        it('should call service.logEvent with correct dto', async () => {
            const dto: SafetyEventDto = {
                ecuId: 'ECU123',
                eventType: 'FLASH_SUCCESS',
                success: true,
                message: 'Everything fine'
            };
            await controller.logEvent(dto);
            expect(service.logEvent).toHaveBeenCalledWith(dto);
        });
    });
});
