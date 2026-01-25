import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { VehiclesController } from './vehicles.controller';
import { VehiclesService } from './vehicles.service';
import { Vehicle } from './entities/vehicle.entity';
import { AuthModule } from '../auth';

@Module({
    imports: [TypeOrmModule.forFeature([Vehicle]), AuthModule],
    controllers: [VehiclesController],
    providers: [VehiclesService],
    exports: [VehiclesService],
})
export class VehiclesModule {}
