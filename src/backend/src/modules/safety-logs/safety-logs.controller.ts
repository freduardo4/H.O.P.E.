import { Controller, Post, Body, HttpCode, HttpStatus } from '@nestjs/common';
import { SafetyLogsService } from './safety-logs.service';
import { ValidateFlashDto } from './dto/validate-flash.dto';
import { SafetyEventDto } from './dto/safety-event.dto';

@Controller('safety')
export class SafetyLogsController {
    constructor(private readonly safetyLogsService: SafetyLogsService) { }

    @Post('validate')
    @HttpCode(HttpStatus.OK)
    async validateFlash(@Body() dto: ValidateFlashDto) {
        return this.safetyLogsService.validateFlash(dto);
    }

    @Post('telemetry')
    @HttpCode(HttpStatus.CREATED)
    async logEvent(@Body() dto: SafetyEventDto) {
        return this.safetyLogsService.logEvent(dto);
    }
}
