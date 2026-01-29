
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AssetsController } from './assets.controller';
import { AssetsService } from './assets.service';
import { License } from '../marketplace/entities/license.entity';

@Module({
    imports: [
        TypeOrmModule.forFeature([License]) // Access License entity for fleet stats
    ],
    controllers: [AssetsController],
    providers: [AssetsService],
})
export class AssetsModule { }
