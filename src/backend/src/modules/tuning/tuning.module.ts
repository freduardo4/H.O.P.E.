
import { Module } from '@nestjs/common';
import { TuningController } from './tuning.controller';
import { TuningService } from './tuning.service';

@Module({
    controllers: [TuningController],
    providers: [TuningService],
})
export class TuningModule { }
