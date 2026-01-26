import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ECUCalibrationsController } from './ecu-calibrations.controller';
import { ECUCalibrationsService } from './ecu-calibrations.service';
import { ECUCalibration } from './entities/ecu-calibration.entity';
import { AuthModule } from '../auth';

@Module({
    imports: [TypeOrmModule.forFeature([ECUCalibration]), AuthModule],
    controllers: [ECUCalibrationsController],
    providers: [ECUCalibrationsService],
    exports: [ECUCalibrationsService],
})
export class ECUCalibrationsModule { }
