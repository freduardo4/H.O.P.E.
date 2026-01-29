
import { Module } from '@nestjs/common';
import { TuningController } from './tuning.controller';
import { TuningService } from './tuning.service';
import { AfrService } from './afr.service';

@Module({
    controllers: [TuningController],
    providers: [TuningService, AfrService],
})
export class TuningModule { }
