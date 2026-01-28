import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { SafetyLogsService } from './safety-logs.service';
import { SafetyLog } from './entities/safety-log.entity';

describe('SafetyLogsService', () => {
    let service: SafetyLogsService;
    let repository: Repository<SafetyLog>;

    const mockRepository = {
        create: jest.fn().mockImplementation((dto) => dto),
        save: jest.fn().mockImplementation((log) => Promise.resolve({ id: 'uuid', ...log })),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                SafetyLogsService,
                {
                    provide: getRepositoryToken(SafetyLog),
                    useValue: mockRepository,
                },
            ],
        }).compile();

        service = module.get<SafetyLogsService>(SafetyLogsService);
        repository = module.get<Repository<SafetyLog>>(getRepositoryToken(SafetyLog));
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    describe('validateFlash', () => {
        it('should allow valid flash operation', async () => {
            const result = await service.validateFlash({ ecuId: 'ECU_VALID', voltage: 12.5 });
            console.log('Valid Result:', result);
            expect(result.allowed).toBe(true);
        });

        it('should deny flash with low voltage', async () => {
            const result = await service.validateFlash({ ecuId: 'ECU_VALID', voltage: 11.0 });
            console.log('Low Voltage Result:', result);
            expect(result.allowed).toBe(false);
            expect(result.reason).toBeDefined();
        });

        it('should deny flash for blacklisted ECU', async () => {
            const result = await service.validateFlash({ ecuId: 'UNSTABLE_BATCH_99', voltage: 13.0 });
            console.log('Blacklist Result:', result);
            expect(result.allowed).toBe(false);
            expect(result.reason).toBeDefined();
        });
    });

    describe('logEvent', () => {
        it('should save log to repository', async () => {
            const dto = {
                eventType: 'FLASH_ATTEMPT',
                ecuId: 'ECU1',
                voltage: 12.5,
                success: true,
                message: 'OK'
            };

            const result = await service.logEvent(dto);
            console.log('Log Result:', result);

            expect(mockRepository.create).toHaveBeenCalled();
            expect(mockRepository.save).toHaveBeenCalled();
            expect(result.ecuId).toBe('ECU1');
        });
    });
});
