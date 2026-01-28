
import { Controller, Post, Body, HttpCode, HttpStatus, Get, Query } from '@nestjs/common';
import { TuningService } from './tuning.service';
import { AfrService } from './afr.service';
import { OptimizeRequestDto, OptimizeResponseDto } from './tuning.dto';

@Controller('tuning')
export class TuningController {
    constructor(
        private readonly tuningService: TuningService,
        private readonly afrService: AfrService
    ) { }

    @Post('optimize')
    @HttpCode(HttpStatus.OK)
    async optimize(@Body() optimizeDto: OptimizeRequestDto): Promise<OptimizeResponseDto> {
        return await this.tuningService.optimizeMap(optimizeDto);
    }

    @Get('afr-targets')
    getAfrTargets(@Query('rows') rows?: number, @Query('cols') cols?: number): number[][] {
        return this.afrService.generateDefaultMap(rows || 16, cols || 16);
    }
}
