import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { SafetyLog } from './entities/safety-log.entity';
import { SafetyEventDto } from './dto/safety-event.dto';
import { ValidateFlashDto } from './dto/validate-flash.dto';

@Injectable()
export class SafetyLogsService {
    private readonly logger = new Logger(SafetyLogsService.name);

    constructor(
        @InjectRepository(SafetyLog)
        private safetyLogRepository: Repository<SafetyLog>,
    ) { }

    async validateFlash(dto: ValidateFlashDto): Promise<{ allowed: boolean; reason?: string }> {
        // Enforce cloud-side safety policy

        // 1. Voltage Check
        const MIN_CLOUD_VOLTAGE = 12.0; // Slightly lower than desktop strict check (12.5V or 13.0V) to allow marginal cases if desktop allows
        if (dto.voltage < MIN_CLOUD_VOLTAGE) {
            this.logger.warn(`Flash denied for ECU ${dto.ecuId}: Low Voltage (${dto.voltage}V)`);
            return { allowed: false, reason: 'Cloud detected unstable voltage telemetry' };
        }

        // 2. Blacklist Check (simulation)
        const BLACKLISTED_ECUS = ['RECALLED_ECU_V1', 'UNSTABLE_BATCH_99'];
        if (BLACKLISTED_ECUS.includes(dto.ecuId)) {
            this.logger.warn(`Flash denied for ECU ${dto.ecuId}: Hardware Blacklisted`);
            return { allowed: false, reason: 'ECU Hardware ID is blacklisted by safety bulletin' };
        }

        this.logger.log(`Flash authorized for ECU ${dto.ecuId} at ${dto.voltage}V`);
        return { allowed: true };
    }

    async logEvent(dto: SafetyEventDto): Promise<SafetyLog> {
        const log = this.safetyLogRepository.create({
            ...dto,
            timestamp: new Date(),
        });

        const saved = await this.safetyLogRepository.save(log);
        this.logger.log(`Safety event logged: ${dto.eventType} for ${dto.ecuId} (Success: ${dto.success})`);
        return saved;
    }
}
