import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ReportsController } from './reports.controller';
import { ReportsService } from './reports.service';
import { Report } from './entities/report.entity';
import { AuthModule } from '../auth';

@Module({
    imports: [TypeOrmModule.forFeature([Report]), AuthModule],
    controllers: [ReportsController],
    providers: [ReportsService],
    exports: [ReportsService],
})
export class ReportsModule { }
